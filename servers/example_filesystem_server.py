# example_filesystem_server.py
"""
Example MCP server that provides filesystem operations.
Save this as a separate file and reference it in servers.yaml.
"""

import os
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Filesystem")

@mcp.tool()
def list_directory(path: str) -> str:
    """List contents of a directory."""
    try:
        files = os.listdir(path)
        return json.dumps(files, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')