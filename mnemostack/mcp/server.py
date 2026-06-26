import logging

from mcp.server.fastmcp import FastMCP

from mnemostack.config.settings import settings

log = logging.getLogger(__name__)

mcp = FastMCP("mnemostack")

# Register all tools (import-as-side-effect; tools attach to `mcp` on load).
from mnemostack.mcp import tools  # noqa: F401, E402


def run() -> None:
    """Run the MCP server until shutdown, then stop the file watcher and close
    index/DB handles. The watcher is started lazily by index_project, so there's
    nothing to start here — only to tear down cleanly on exit."""
    from mnemostack.core.state import state

    try:
        mcp.run(transport=settings.server.transport)
    finally:
        # Guard teardown so a failing handle can't mask the real shutdown error.
        try:
            state.close()
        except Exception:
            log.exception("error during state teardown")
