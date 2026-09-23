"""SQLite index over the local run logs written by app/local_trace.py.

The run directories (events.jsonl + summary.json + blobs/) are the source of
truth; this DB is a derived, rebuildable index for comparing many runs. Rows
hold hashes + previews only — full payloads stay in the blob files.
"""

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    job_id TEXT,
    started_at TEXT,
    duration_ms INTEGER,
    status TEXT,
    error TEXT,
    industry TEXT,
    ticker TEXT,
    commit_hash TEXT,
    sonnet_model TEXT,
    haiku_model TEXT,
    llm_calls INTEGER,
    tool_calls INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error_count INTEGER,
    dir TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    ts TEXT,
    event TEXT,
    kind TEXT,
    name TEXT,
    node TEXT,
    node_path TEXT,
    model TEXT,
    run_uuid TEXT,
    parent_uuid TEXT,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read INTEGER,
    cache_creation INTEGER,
    input_refs TEXT,
    output_ref TEXT,
    input_preview TEXT,
    output_preview TEXT,
    truncated INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_node ON events(node);
CREATE INDEX IF NOT EXISTS idx_events_kind_name ON events(kind, name);
CREATE INDEX IF NOT EXISTS idx_events_error ON events(error);
CREATE VIEW IF NOT EXISTS node_totals AS
SELECT
    run_id,
    COALESCE(node, '(none)') AS node,
    SUM(kind = 'llm' AND event IN ('end', 'error')) AS llm_calls,
    SUM(kind = 'tool' AND event IN ('end', 'error')) AS tool_calls,
    COALESCE(SUM(input_tokens), 0) AS input_tokens,
    COALESCE(SUM(output_tokens), 0) AS output_tokens,
    COALESCE(SUM(latency_ms), 0) AS latency_ms
FROM events
WHERE kind IN ('llm', 'tool')
GROUP BY run_id, COALESCE(node, '(none)');
"""

# Call-counting rule shared with app/local_trace.py's summary totals: a call is
# counted once, on its terminal event (end or error).
TERMINAL_EVENTS = ("end", "error")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def _read_events(events_path: Path) -> list[dict]:
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed line in %s", events_path)  # e.g. crash mid-write
    return events


def _event_row(run_id: str, e: dict) -> tuple:
    usage = e.get("usage") or {}
    return (
        run_id,
        e.get("ts"),
        e.get("event"),
        e.get("kind"),
        e.get("name"),
        e.get("node"),
        e.get("node_path"),
        e.get("model"),
        e.get("run_uuid"),
        e.get("parent_uuid"),
        e.get("latency_ms"),
        usage.get("input"),
        usage.get("output"),
        usage.get("cache_read"),
        usage.get("cache_creation"),
        json.dumps(e.get("input_refs") or []),
        e.get("output_ref"),
        e.get("input_preview"),
        e.get("output_preview"),
        1 if e.get("truncated") else 0,
        e.get("error"),
    )


def index_run(db_path: str | Path, run_dir: str | Path) -> None:
    """Upsert one run. Totals are recomputed from events so a run whose
    summary.json was never written (crash) is still indexed, and re-indexing
    (or a full rebuild) always yields identical rows."""
    run_dir = Path(run_dir)
    run_id = run_dir.name
    events = _read_events(run_dir / "events.jsonl")

    summary: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    tags = summary.get("tags") or {}
    inputs = summary.get("input") or {}

    llm_calls = sum(1 for e in events if e.get("kind") == "llm" and e.get("event") in TERMINAL_EVENTS)
    tool_calls = sum(1 for e in events if e.get("kind") == "tool" and e.get("event") in TERMINAL_EVENTS)
    input_tokens = sum((e.get("usage") or {}).get("input") or 0 for e in events)
    output_tokens = sum((e.get("usage") or {}).get("output") or 0 for e in events)
    error_count = sum(1 for e in events if e.get("event") == "error")

    run_row = (
        run_id,
        summary.get("job_id"),
        summary.get("started_at") or (events[0].get("ts") if events else None),
        summary.get("duration_ms"),
        summary.get("status") or "incomplete",
        summary.get("error"),
        inputs.get("industry"),
        inputs.get("ticker"),
        tags.get("commit_hash"),
        tags.get("sonnet_model"),
        tags.get("haiku_model"),
        llm_calls,
        tool_calls,
        input_tokens,
        output_tokens,
        error_count,
        str(run_dir),
    )

    conn = connect(db_path)
    try:
        with conn:  # one transaction: delete + insert, so re-indexing never duplicates rows
            conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", run_row)
            conn.executemany(
                "INSERT INTO events (run_id, ts, event, kind, name, node, node_path, model, run_uuid,"
                " parent_uuid, latency_ms, input_tokens, output_tokens, cache_read, cache_creation,"
                " input_refs, output_ref, input_preview, output_preview, truncated, error)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [_event_row(run_id, e) for e in events],
            )
    finally:
        conn.close()


def delete_run(db_path: str | Path, run_id: str) -> None:
    if not Path(db_path).exists():
        return
    conn = connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    finally:
        conn.close()


def rebuild(db_path: str | Path, trace_dir: str | Path) -> int:
    """Drop and recreate the index from every run directory under `trace_dir`."""
    path = Path(db_path)
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    count = 0
    trace_dir = Path(trace_dir)
    if trace_dir.is_dir():
        for run_dir in sorted(p for p in trace_dir.iterdir() if p.is_dir()):
            index_run(path, run_dir)
            count += 1
    else:
        connect(path).close()
    return count


def _table(headers: list[str], rows: list[tuple]) -> str:
    cells = [headers] + [["" if v is None else str(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def report(db_path: str | Path, top: int = 5) -> str:
    conn = connect(db_path)
    try:
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        per_node = conn.execute(
            "SELECT node, SUM(llm_calls), SUM(tool_calls), SUM(input_tokens), SUM(output_tokens),"
            " ROUND(SUM(latency_ms) / 1000.0, 1) FROM node_totals GROUP BY node ORDER BY SUM(input_tokens) DESC"
        ).fetchall()
        tool_errors = conn.execute(
            "SELECT name, SUM(event = 'error'), COUNT(*), ROUND(100.0 * SUM(event = 'error') / COUNT(*), 1)"
            " FROM events WHERE kind = 'tool' AND event IN ('end', 'error') GROUP BY name ORDER BY 2 DESC"
        ).fetchall()
        costliest = conn.execute(
            "SELECT run_id, status, COALESCE(industry, ticker), llm_calls, tool_calls, input_tokens,"
            " output_tokens, ROUND(COALESCE(duration_ms, 0) / 1000.0, 1) FROM runs"
            " ORDER BY input_tokens + output_tokens DESC LIMIT ?",
            (top,),
        ).fetchall()
    finally:
        conn.close()

    return "\n\n".join(
        [
            f"{n_runs} run(s) indexed",
            "Per-node totals (all runs)\n"
            + _table(["node", "llm", "tool", "in_tok", "out_tok", "latency_s"], per_node),
            "Tool error rate\n" + _table(["tool", "errors", "calls", "error_%"], tool_errors),
            f"Top {top} runs by tokens\n"
            + _table(["run", "status", "input", "llm", "tool", "in_tok", "out_tok", "secs"], costliest),
        ]
    )
