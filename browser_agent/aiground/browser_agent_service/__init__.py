# -*- coding: utf-8 -*-
"""
browser_agent_service

Open-source browser agent service implementation (pure FastAPI + MCP).
This package is intentionally UI-free (no gradio/grpages dependencies).
"""

from .server import BrowserAgentServer

__all__ = ["BrowserAgentServer"]


