"""Anima MCP Server - Model Context Protocol integration.

Exposes Anima capabilities to Claude Code and other MCP clients.
"""

from .server import mcp, run_server

__all__ = ["mcp", "run_server"]
