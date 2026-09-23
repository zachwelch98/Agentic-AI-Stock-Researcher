"""Local, offline run logging via a passive LangChain callback handler.

Purely observational: it records events the graph already emits (LLM calls,
tool calls, graph-node runs) and never makes an LLM/tool call, adds a graph
node, or alters a prompt. Enabled by LOCAL_TRACE_DIR (see app/config.py).

Per run, under `<LOCAL_TRACE_DIR>/<utc-timestamp>_<job_id>/`:
    events.jsonl         small append-only event log (previews + blob refs)
    blobs/<sha256>.json  full payloads, content-addressed so identical content
                         (e.g. a ReAct conversation resent every turn) is stored once
    summary.json         written at job end: status, tags, per-node totals

The SQLite index (app/trace_index.py) is built from these files.
"""

import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage, message_to_dict

from app import trace_index
from app.config import Settings

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 300
# Internal node names inside langgraph.prebuilt.create_react_agent; events under
# them are attributed to the enclosing researcher node instead.
_INTERNAL_NODES = {"agent", "tools", "generate_structured_response"}
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _default(obj: Any) -> Any:
    if isinstance(obj, BaseMessage):
        return message_to_dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False, sort_keys=True)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or f"[{block.get('type', 'block')}]")
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def _message_preview(message: BaseMessage) -> str:
    text = _content_text(message.content)
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        text += f" [tool_call] {call.get('name')} {_dumps(call.get('args'))}"
    return f"[{message.type}] {text}".strip()


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


def _usage_from(response: Any) -> dict:
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    for generations in getattr(response, "generations", []) or []:
        for gen in generations:
            um = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if not um:
                continue
            details = um.get("input_token_details") or {}
            usage["input"] += um.get("input_tokens", 0) or 0
            usage["output"] += um.get("output_tokens", 0) or 0
            usage["cache_read"] += details.get("cache_read", 0) or 0
            usage["cache_creation"] += details.get("cache_creation", 0) or 0
    return usage


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=2
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _safe(fn):
    """Tracing must never fail a job: swallow + log anything a hook raises."""

    @functools.wraps(fn)
    async def wrapper(self, *args, **kwargs):
        try:
            return await fn(self, *args, **kwargs)
        except Exception:
            logger.warning("Local trace handler failed in %s", fn.__name__, exc_info=True)

    return wrapper


class LocalTraceHandler(AsyncCallbackHandler):
    def __init__(
        self,
        run_dir: Path,
        *,
        job_id: str,
        industry: str | None = None,
        ticker: str | None = None,
        max_blob_bytes: int = 200_000,
        secrets: list[str] | None = None,
        db_path: str | None = None,
        sonnet_model: str | None = None,
        haiku_model: str | None = None,
    ):
        self.run_dir = run_dir
        self.job_id = job_id
        self.industry = industry
        self.ticker = ticker
        self.max_blob_bytes = max_blob_bytes
        self.db_path = db_path or None
        self.sonnet_model = sonnet_model
        self.haiku_model = haiku_model
        # Very short "secrets" would over-redact ordinary text.
        self._secrets = [s for s in (secrets or []) if s and len(s) >= 8]

        self._lock = threading.Lock()
        self._started_at = _now()
        self._t0 = time.monotonic()
        self._parent: dict[UUID, UUID | None] = {}
        self._node_name: dict[UUID, str] = {}  # only for traced graph-node chain runs
        self._started: dict[UUID, tuple[float, str, str]] = {}  # run_uuid -> (monotonic start, kind, name)
        self._blobs_seen: set[str] = set()
        self._by_node: dict[str, dict] = {}
        self._totals = {"llm_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
        self._finished = False

        (run_dir / "blobs").mkdir(parents=True, exist_ok=True)

    # ---- storage helpers -------------------------------------------------

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        return text

    def _put(self, obj: Any) -> tuple[str, bool]:
        """Store `obj` as a content-addressed blob; returns (sha256, truncated)."""
        text = self._redact(_dumps(obj))
        raw = text.encode()
        truncated = len(raw) > self.max_blob_bytes
        if truncated:
            text = _dumps(
                {"_truncated": True, "original_bytes": len(raw), "text": raw[: self.max_blob_bytes].decode("utf-8", "ignore")}
            )
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha not in self._blobs_seen:
            path = self.run_dir / "blobs" / f"{sha}.json"
            if not path.exists():
                _atomic_write_text(path, text)
            self._blobs_seen.add(sha)
        return sha, truncated

    def _emit(self, record: dict) -> None:
        line = json.dumps({k: v for k, v in record.items() if v is not None}, ensure_ascii=False)
        with self._lock:
            with open(self.run_dir / "events.jsonl", "a") as f:
                f.write(line + "\n")

    def _node_path(self, run_id: UUID) -> list[str]:
        path: list[str] = []
        current: UUID | None = run_id
        while current is not None:
            if current in self._node_name:
                path.append(self._node_name[current])
            current = self._parent.get(current)
        return list(reversed(path))

    def _effective_node(self, path: list[str]) -> str | None:
        for name in reversed(path):
            if name not in _INTERNAL_NODES:
                return name
        return path[-1] if path else None

    def _agg(self, node: str | None) -> dict:
        return self._by_node.setdefault(
            node or "(none)",
            {"llm_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0},
        )

    def _start_event(self, kind: str, name: str, run_id: UUID, parent_id: UUID | None, **fields: Any) -> None:
        self._parent[run_id] = parent_id
        self._started[run_id] = (time.monotonic(), kind, name)
        path = self._node_path(run_id)
        self._emit(
            {
                "ts": _now(),
                "event": "start",
                "kind": kind,
                "name": name,
                "run_uuid": str(run_id),
                "parent_uuid": str(parent_id) if parent_id else None,
                "node": self._effective_node(path),
                "node_path": "/".join(path) or None,
                **fields,
            }
        )

    def _end_event(
        self,
        event: str,
        kind: str,
        run_id: UUID,
        parent_id: UUID | None,
        name: str,
        *,
        model: str | None = None,
        usage: dict | None = None,
        output: Any = None,
        output_preview: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        started = self._started.pop(run_id, None)
        latency_ms = int((time.monotonic() - started[0]) * 1000) if started else None
        name = started[2] if started else name
        path = self._node_path(run_id)
        node = self._effective_node(path)

        output_ref = truncated = None
        if output is not None:
            output_ref, truncated = self._put(output)
        if error is not None:
            self._totals["errors"] += 1

        if kind in ("llm", "tool"):
            agg = self._agg(node)
            agg[f"{kind}_calls"] += 1
            agg["latency_ms"] += latency_ms or 0
            self._totals[f"{kind}_calls"] += 1
            if usage:
                agg["input_tokens"] += usage["input"]
                agg["output_tokens"] += usage["output"]
                self._totals["input_tokens"] += usage["input"]
                self._totals["output_tokens"] += usage["output"]

        self._emit(
            {
                "ts": _now(),
                "event": event,
                "kind": kind,
                "name": name,
                "run_uuid": str(run_id),
                "parent_uuid": str(parent_id) if parent_id else None,
                "node": node,
                "node_path": "/".join(path) or None,
                "model": model,
                "latency_ms": latency_ms,
                "usage": usage,
                "output_ref": output_ref,
                "output_preview": self._redact(output_preview) if output_preview is not None else None,
                "truncated": truncated or None,
                "error": self._redact(f"{type(error).__name__}: {error}") if error is not None else None,
            }
        )

    @staticmethod
    def _model_name(metadata: dict | None, kwargs: dict) -> str | None:
        return (kwargs.get("invocation_params") or {}).get("model") or (metadata or {}).get("ls_model_name")

    # ---- chain (graph nodes) ---------------------------------------------

    @staticmethod
    def _chain_name(serialized: dict | None, kwargs: dict) -> str:
        return kwargs.get("name") or (serialized or {}).get("name") or ""

    def _is_traced_chain(self, name: str, parent_run_id: UUID | None, metadata: dict | None) -> bool:
        """The root graph run, plus each graph-node run (langgraph tags a node's
        own run with metadata.langgraph_node == its name). Everything else
        (channel writes, runnable sequences, ...) is noise."""
        return parent_run_id is None or (metadata or {}).get("langgraph_node") == name

    @_safe
    async def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        name = self._chain_name(serialized, kwargs)
        self._parent[run_id] = parent_run_id
        if not self._is_traced_chain(name, parent_run_id, metadata):
            return
        if parent_run_id is not None:
            self._node_name[run_id] = name
        if name in _INTERNAL_NODES:
            # Timing only: these steps' state is the growing ReAct message list, which
            # the llm events already record message-by-message (deduped by hash).
            self._start_event("chain", name, run_id, parent_run_id)
            return
        ref, truncated = self._put(inputs)
        self._start_event(
            "chain", name, run_id, parent_run_id, input_refs=[ref], input_preview=_clip(self._redact(_dumps(inputs))),
            truncated=truncated or None,
        )

    @_safe
    async def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        if run_id not in self._started or self._started[run_id][1] != "chain":
            return
        name = self._node_name.get(run_id, "LangGraph")
        if name in _INTERNAL_NODES:
            self._end_event("end", "chain", run_id, parent_run_id, name)
            return
        self._end_event(
            "end", "chain", run_id, parent_run_id, name, output=outputs, output_preview=_clip(_dumps(outputs))
        )

    @_safe
    async def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        if run_id not in self._started or self._started[run_id][1] != "chain":
            return
        name = self._node_name.get(run_id, "LangGraph")
        self._end_event("error", "chain", run_id, parent_run_id, name, error=error)

    # ---- LLM ---------------------------------------------------------------

    @_safe
    async def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs
    ):
        flat: list[BaseMessage] = [m for batch in messages for m in batch]
        blobs_before = len(self._blobs_seen)
        refs = [self._put(m)[0] for m in flat]
        new = len(self._blobs_seen) - blobs_before  # messages not already stored earlier in this run
        preview = f"({len(flat)} messages; {new} new) " + (_message_preview(flat[-1]) if flat else "")
        self._start_event(
            "llm",
            self._chain_name(serialized, kwargs) or "chat_model",
            run_id,
            parent_run_id,
            model=self._model_name(metadata, kwargs),
            input_refs=refs,
            input_preview=_clip(self._redact(preview)),
        )

    @_safe
    async def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        if run_id in self._started:  # already recorded via on_chat_model_start
            return
        refs = [self._put(p)[0] for p in prompts]
        self._start_event(
            "llm",
            self._chain_name(serialized, kwargs) or "llm",
            run_id,
            parent_run_id,
            model=self._model_name(metadata, kwargs),
            input_refs=refs,
            input_preview=_clip(self._redact(prompts[-1] if prompts else "")),
        )

    @_safe
    async def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        generation = next((g for gens in response.generations for g in gens), None)
        message = getattr(generation, "message", None)
        output: Any = message if message is not None else getattr(generation, "text", "")
        preview = _message_preview(message) if message is not None else str(output)
        self._end_event(
            "end",
            "llm",
            run_id,
            parent_run_id,
            "chat_model",
            model=(getattr(message, "response_metadata", None) or {}).get("model_name")
            or (getattr(message, "response_metadata", None) or {}).get("model"),
            usage=_usage_from(response),
            output=output,
            output_preview=_clip(preview),
        )

    @_safe
    async def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end_event("error", "llm", run_id, parent_run_id, "chat_model", error=error)

    # ---- tools -------------------------------------------------------------

    @_safe
    async def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, inputs=None, **kwargs
    ):
        payload = inputs if inputs is not None else input_str
        ref, truncated = self._put(payload)
        self._start_event(
            "tool",
            kwargs.get("name") or (serialized or {}).get("name") or "tool",
            run_id,
            parent_run_id,
            input_refs=[ref],
            input_preview=_clip(self._redact(_dumps(payload))),
            truncated=truncated or None,
        )

    @_safe
    async def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        text = _message_preview(output) if isinstance(output, BaseMessage) else _dumps(output)
        self._end_event(
            "end", "tool", run_id, parent_run_id, "tool", output=output, output_preview=_clip(text)
        )

    @_safe
    async def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end_event("error", "tool", run_id, parent_run_id, "tool", error=error)

    # ---- job end -----------------------------------------------------------

    def finish(self, status: str, error: str | None = None) -> None:
        """Write summary.json and refresh the SQLite index. Never raises."""
        if self._finished:
            return
        self._finished = True
        try:
            summary = {
                "run_id": self.run_dir.name,
                "job_id": self.job_id,
                "input": {"industry": self.industry, "ticker": self.ticker},
                "status": status,
                "error": self._redact(error) if error else None,
                "started_at": self._started_at,
                "duration_ms": int((time.monotonic() - self._t0) * 1000),
                "tags": {
                    "commit_hash": _git_commit(),
                    "sonnet_model": self.sonnet_model,
                    "haiku_model": self.haiku_model,
                },
                "totals": self._totals,
                "by_node": self._by_node,
                "unique_blobs": len(self._blobs_seen),
            }
            _atomic_write_text(self.run_dir / "summary.json", json.dumps(summary, indent=2))
        except Exception:
            logger.warning("Failed to write trace summary for %s", self.run_dir, exc_info=True)
        if self.db_path:
            try:
                trace_index.index_run(self.db_path, self.run_dir)
            except Exception:
                logger.warning("Failed to index trace run %s", self.run_dir, exc_info=True)


def prune_old_runs(trace_dir: Path, keep: int, db_path: str | None = None) -> list[str]:
    """Delete the oldest run directories (and their index rows) beyond `keep`.
    Run dir names start with a UTC timestamp, so name order is chronological."""
    if keep <= 0 or not trace_dir.is_dir():
        return []
    runs = sorted(p for p in trace_dir.iterdir() if p.is_dir())
    removed = []
    for old in runs[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
        if db_path:
            trace_index.delete_run(db_path, old.name)
        removed.append(old.name)
    return removed


def start_run_trace(
    job_id: str, industry: str | None, ticker: str | None, settings: Settings
) -> LocalTraceHandler | None:
    """Create the run dir + handler for a job, or None when tracing is off or
    setup fails (tracing must never block or fail the job)."""
    if not settings.local_trace_dir:
        return None
    try:
        trace_dir = Path(settings.local_trace_dir)
        run_dir = trace_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{job_id}"
        handler = LocalTraceHandler(
            run_dir,
            job_id=job_id,
            industry=industry,
            ticker=ticker,
            max_blob_bytes=settings.local_trace_max_blob_bytes,
            secrets=[
                settings.anthropic_api_key,
                settings.finnhub_api_key,
                os.environ.get("LANGCHAIN_API_KEY", ""),
            ],
            db_path=settings.local_trace_db,
            sonnet_model=settings.sonnet_model,
            haiku_model=settings.haiku_model,
        )
        prune_old_runs(trace_dir, settings.local_trace_keep_runs, settings.local_trace_db or None)
        return handler
    except Exception:
        logger.warning("Could not start local trace for job %s", job_id, exc_info=True)
        return None
