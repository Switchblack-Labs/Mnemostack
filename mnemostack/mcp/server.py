from mcp.server.fastmcp import FastMCP

from mnemostack.config.settings import settings

mcp = FastMCP("mnemostack")

# Register all tools (import-as-side-effect; tools attach to `mcp` on load).
from mnemostack.mcp import tools  # noqa: F401, E402


def run() -> None:
    mcp.run(transport=settings.server.transport)
