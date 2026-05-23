from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mnemostack")

# Register all tools
from mnemostack.mcp import tools  # noqa: F401, E402


def run():
    mcp.run(transport="stdio")
