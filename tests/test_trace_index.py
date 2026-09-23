"""SQLite index over local run logs (app/trace_index.py). Runs are produced by
the real handler via run_job (fake chat model, no network) so the index is
checked against genuine handler output, and against summary.json."""

import sqlite3
from pathlib import Path

from app import jobs, trace_index
from app.config import Settings
from tests.test_local_trace import (  # noqa: F401  (settings is a fixture)
    _failing_graph,
    _react_graph,
    _run,
    _scripted_model,
    _summary,
    settings,
)


def _rows(db: str, sql: str) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


async def test_run_is_indexed_automatically_at_job_end(settings):
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")

    run_id = Path(job.trace_dir).name
    (row,) = _rows(
        settings.local_trace_db,
        "SELECT status, ticker, llm_calls, tool_calls, input_tokens, output_tokens, commit_hash, sonnet_model "
        f"FROM runs WHERE run_id = '{run_id}'",
    )
    assert row[:6] == ("done", "ZION", 2, 1, 260, 22)
    assert row[7] == settings.sonnet_model
    n_lines = len((Path(job.trace_dir) / "events.jsonl").read_text().splitlines())
    assert _rows(settings.local_trace_db, f"SELECT COUNT(*) FROM events WHERE run_id = '{run_id}'")[0][0] == n_lines


async def test_node_totals_view_matches_summary_json(settings):
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")

    summary = _summary(job)
    rows = _rows(
        settings.local_trace_db,
        "SELECT node, llm_calls, tool_calls, input_tokens, output_tokens, latency_ms FROM node_totals",
    )
    view = {r[0]: r[1:] for r in rows}
    assert set(view) == set(summary["by_node"])
    for node, agg in summary["by_node"].items():
        assert view[node] == (
            agg["llm_calls"],
            agg["tool_calls"],
            agg["input_tokens"],
            agg["output_tokens"],
            agg["latency_ms"],
        )


async def test_reindexing_is_idempotent(settings):
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")
    before = _rows(settings.local_trace_db, "SELECT COUNT(*) FROM events")[0][0]

    trace_index.index_run(settings.local_trace_db, job.trace_dir)
    trace_index.index_run(settings.local_trace_db, job.trace_dir)

    assert _rows(settings.local_trace_db, "SELECT COUNT(*) FROM runs")[0][0] == 1
    assert _rows(settings.local_trace_db, "SELECT COUNT(*) FROM events")[0][0] == before


async def test_rebuild_from_files_reproduces_the_same_rows(settings, tmp_path):
    await _run(_react_graph(_scripted_model()), ticker="ZION")
    await _run(_failing_graph(), ticker="ZION")

    runs_sql = "SELECT run_id, status, llm_calls, tool_calls, input_tokens, error_count FROM runs ORDER BY run_id"
    events_sql = "SELECT run_id, ts, event, kind, name, node, input_tokens FROM events ORDER BY run_id, ts, id"
    before = (_rows(settings.local_trace_db, runs_sql), _rows(settings.local_trace_db, events_sql))
    assert len(before[0]) == 2

    count = trace_index.rebuild(settings.local_trace_db, settings.local_trace_dir)

    assert count == 2
    after = (_rows(settings.local_trace_db, runs_sql), _rows(settings.local_trace_db, events_sql))
    assert after == before


async def test_failed_run_and_tool_error_rate_are_queryable(settings):
    await _run(_failing_graph(), ticker="ZION")

    (row,) = _rows(settings.local_trace_db, "SELECT status, error, error_count FROM runs")
    assert row[0] == "failed" and "upstream exploded" in row[1] and row[2] >= 1


async def test_run_without_summary_is_still_indexed(settings):
    """A crash before summary.json is written must not hide the run."""
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")
    summary_path = Path(job.trace_dir) / "summary.json"
    summary_path.unlink()

    trace_index.rebuild(settings.local_trace_db, settings.local_trace_dir)

    (row,) = _rows(settings.local_trace_db, "SELECT status, llm_calls, input_tokens FROM runs")
    assert row == ("incomplete", 2, 260)


async def test_malformed_event_line_is_skipped(settings):
    job = await _run(_react_graph(_scripted_model()), ticker="ZION")
    events_path = Path(job.trace_dir) / "events.jsonl"
    good = len(events_path.read_text().splitlines())
    events_path.write_text(events_path.read_text() + '{"ts": "truncated mid-wri')  # crash mid-write

    trace_index.index_run(settings.local_trace_db, job.trace_dir)

    assert _rows(settings.local_trace_db, "SELECT COUNT(*) FROM events")[0][0] == good


async def test_corrupt_db_does_not_fail_the_job(tmp_path, monkeypatch):
    db = tmp_path / "traces.db"
    db.write_bytes(b"this is not a sqlite database" * 100)
    s = Settings(local_trace_dir=str(tmp_path / "runs"), local_trace_db=str(db))
    monkeypatch.setattr(jobs, "get_settings", lambda: s)

    job = await _run(_react_graph(_scripted_model()), ticker="ZION")

    assert job.status.value == "done"
    assert _summary(job)["status"] == "done"  # files are still the source of truth


async def test_report_renders_per_node_tool_errors_and_costliest_runs(settings):
    await _run(_react_graph(_scripted_model()), ticker="ZION")

    text = trace_index.report(settings.local_trace_db, top=3)

    assert "1 run(s) indexed" in text
    assert "news_researcher" in text
    assert "get_price" in text
    assert "Top 3 runs by tokens" in text

