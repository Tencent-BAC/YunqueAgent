"""FastMCP - A more ergonomic interface for MCP servers."""

import importlib.metadata

from .server import Context, FastMCP
from .utilities.types import Image

try:
    __version__ = importlib.metadata.version("mcp")
except importlib.metadata.PackageNotFoundError:  # vendored usage (not installed as a dist)
    __version__ = "unknown"
__all__ = ["FastMCP", "Context", "Image"]
