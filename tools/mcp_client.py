import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def repo_tools(repo: Path):
    """Start the MCP server for a repo and yield the tools it advertises."""
    client = MultiServerMCPClient({
        "repo": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "tools.mcp_server", str(repo)],
            "cwd": str(PROJECT_ROOT),
            # The MCP client passes only a safe subset of the parent env (PATH, HOME...), never API keys.
            "env": {},
        }
    })
    async with client.session("repo") as session:
        tools: list[BaseTool] = await load_mcp_tools(session)
        yield tools
