"""FastAPI app factory + lifespan.

The lifespan spawns both MCP stdio subprocesses exactly once (persistent
sessions held open by an AsyncExitStack for the app's lifetime — see
tools/mcp_client.py for why the naive MultiServerMCPClient.get_tools() isn't
used here), primes the NASDAQ universe cache, and compiles the research graph.
"""

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.graph.build import build_research_graph
from tools.mcp_client import build_mcp_client, open_persistent_mcp_sessions, set_mcp_tools
from tools.mcp_finance_server import nasdaq_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Priming NASDAQ universe cache...")
    await nasdaq_universe.get_universe()

    async with AsyncExitStack() as stack:
        logger.info("Starting MCP servers (finance + fetch)...")
        mcp_client = build_mcp_client()
        tools = await open_persistent_mcp_sessions(mcp_client, stack)
        set_mcp_tools(tools)
        logger.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])

        app.state.mcp_client = mcp_client
        app.state.mcp_tools = tools
        app.state.research_graph = build_research_graph()
        app.state.background_tasks = set()

        yield

    logger.info("MCP servers shut down.")


def create_app(lifespan_override=None) -> FastAPI:
    """`lifespan_override` lets tests swap in a stub lifespan (a trivial graph,
    no real MCP subprocesses or API calls) while exercising the real HTTP
    contract — routes, schemas, status codes."""
    app = FastAPI(title="Agentic AI Stock Researcher", lifespan=lifespan_override or lifespan)
    app.include_router(router)
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
    return app


app = create_app()
