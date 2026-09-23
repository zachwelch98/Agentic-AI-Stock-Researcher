"""Standalone manual/cron refresh of the NASDAQ ticker-universe cache.

    uv run python scripts/refresh_nasdaq_cache.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.mcp_finance_server import nasdaq_universe  # noqa: E402

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    cache = await nasdaq_universe.refresh_cache()
    rows = cache["rows"]
    print(f"Refreshed NASDAQ universe cache: {len(rows)} rows -> {nasdaq_universe.CACHE_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
