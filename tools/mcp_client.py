"""MultiServerMCPClient wiring for both MCP servers.

`MultiServerMCPClient.get_tools()` starts a brand-new stdio subprocess session
for every single tool call (that's documented behavior, not a bug — see its
docstring: "a new session will be created for each tool call"). That violates
this project's "both servers are spawned once, not per-request" design, so
`open_persistent_mcp_sessions()` instead opens one long-lived session per
server via `client.session(name)`, held open for the app's lifetime by an
`AsyncExitStack` owned by FastAPI's `lifespan` (app/main.py), and loads tools
bound to those persistent sessions. Tools are cached in this module's
singleton; graph node functions pull them from there via `get_mcp_tool()`.
"""

import json
import sys
from contextlib import AsyncExitStack

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def build_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "finance": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "tools.mcp_finance_server.server"],
            },
            "fetch": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server_fetch"],
            },
        }
    )


async def open_persistent_mcp_sessions(client: MultiServerMCPClient, exit_stack: AsyncExitStack) -> list[BaseTool]:
    """Enter one long-lived session per configured server (registered on
    `exit_stack` so both are closed together at shutdown) and load tools bound
    to those persistent sessions — each stdio subprocess is spawned exactly
    once for the app's lifetime, not per tool call."""
    all_tools: list[BaseTool] = []
    for server_name in client.connections:
        session = await exit_stack.enter_async_context(client.session(server_name))
        tools = await load_mcp_tools(session, server_name=server_name)
        all_tools.extend(tools)
    return all_tools


def tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}


_mcp_tools: dict[str, BaseTool] | None = None


def set_mcp_tools(tools: list[BaseTool]) -> None:
    """Called once from FastAPI's lifespan after `open_persistent_mcp_sessions`."""
    global _mcp_tools
    _mcp_tools = tools_by_name(tools)


def get_mcp_tool(name: str) -> BaseTool:
    if _mcp_tools is None:
        raise RuntimeError(
            "MCP tools not initialized — set_mcp_tools() must run in FastAPI's "
            "lifespan (or a test fixture) before any graph node calls get_mcp_tool()."
        )
    return _mcp_tools[name]


def parse_mcp_tool_result(result: object) -> dict:
    """MCP tool results come back from langchain-mcp-adapters as a list of
    content blocks (e.g. [{"type": "text", "text": "<json>"}]), not a parsed
    dict — unwrap that shape into the JSON payload the tool actually returned."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, list):
        text = "".join(
            block.get("text", "") for block in result if isinstance(block, dict) and block.get("type") == "text"
        )
        return json.loads(text)
    raise TypeError(f"Unexpected MCP tool result type: {type(result)!r}")
