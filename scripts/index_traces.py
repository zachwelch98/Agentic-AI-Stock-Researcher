"""Build/refresh the SQLite index over local run logs, and print a report.

    uv run python scripts/index_traces.py                 # index new/changed runs (upsert all)
    uv run python scripts/index_traces.py --rebuild       # drop and recreate from traces/runs/
    uv run python scripts/index_traces.py --report        # per-node cost, tool error rate, costliest runs

The run directories are the source of truth; the DB is derived and rebuildable.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import trace_index  # noqa: E402

logging.basicConfig(level=logging.INFO)

DEFAULT_TRACE_DIR = "traces/runs"
DEFAULT_DB = "traces/traces.db"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--rebuild", action="store_true", help="drop the DB and re-index every run")
    parser.add_argument("--report", action="store_true", help="print summary tables after indexing")
    parser.add_argument("--top", type=int, default=5, help="rows in the costliest-runs table")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if args.rebuild:
        count = trace_index.rebuild(args.db, trace_dir)
    else:
        count = 0
        for run_dir in sorted(p for p in trace_dir.iterdir() if p.is_dir()) if trace_dir.is_dir() else []:
            trace_index.index_run(args.db, run_dir)
            count += 1
    print(f"Indexed {count} run(s) from {trace_dir} -> {args.db}")

    if args.report:
        print()
        print(trace_index.report(args.db, top=args.top))


if __name__ == "__main__":
    main()
