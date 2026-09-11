"""MCP server entry point.

Nothing to tear down any more: the upgrade check opens its graph, uses it, and
closes it within the call. The old teardown existed for a long-lived index and
a file watcher, both of which are gone.
"""

from __future__ import annotations

from mnemostack.config.settings import settings
from mnemostack.mcp.tools import mcp


def run() -> None:
    """Run the MCP server until shutdown."""
    mcp.run(transport=settings.server.transport)
