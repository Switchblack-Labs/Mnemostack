"""MCP server entry point: `mnemostack` with no arguments, over stdio."""

from __future__ import annotations

from mnemostack.mcp.tools import mcp


def run() -> None:
    """Run the MCP server until shutdown."""
    mcp.run()
