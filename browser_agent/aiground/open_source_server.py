# -*- coding: utf-8 -*-
"""
open_source_server

Open-source entrypoint:
- HTTP API: prefixless (root path, no group prefix)
- MCP SSE: mounted at `/mcp` (`/mcp` + `/mcp/messages/`)
"""

import logging
import os
from typing import Any, Dict

import uvicorn
import yaml
from fastapi import FastAPI

from aiground.common.dict_args import DictArgs
from aiground.framework.thirdparty.fastapi_mcp.server import FastApiMCP

from .browser_agent_service import BrowserAgentServer

LOGGER = logging.getLogger(__name__)


def _load_yaml_config(path: str) -> DictArgs:
    path = (path or "").strip()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        return DictArgs(data)
    return DictArgs({})


def create_app(config: DictArgs) -> FastAPI:
    app = FastAPI(title="browser_agent", version="0.1.0")

    # Normalize minimal sections to avoid attribute errors (DictArgs returns {} if missing)
    if not config.get("browser_use_tool"):
        config["browser_use_tool"] = {}
    if not config.get("web_search_tool"):
        config["web_search_tool"] = {}
    if not config.get("trace"):
        config["trace"] = {}

    # Prefer environment variables for secrets.
    # - Serper (Google via serper.dev)
    serper_env = (os.environ.get("SERPER_KEY_ID", "") or "").strip()
    if not serper_env:
        serper_cfg = (config.get("web_search_tool", {}) or {}).get("serper_key_id", "")
        if serper_cfg:
            os.environ["SERPER_KEY_ID"] = str(serper_cfg)

    # - Jina r.jina.ai token
    jina_env = (os.environ.get("JINA_API_KEY", "") or "").strip()
    if jina_env and not (config.get("browser_use_tool", {}) or {}).get("jina_api_key"):
        config["browser_use_tool"]["jina_api_key"] = jina_env

    # Hard requirement: browser_use_tool needs browser/llm config,
    # otherwise /browser_use and /browser_use_agent will not work.
    # Use config/browser_agent.example.yaml as a minimal template.
    missing = []
    bu = config.get("browser_use_tool", {}) or {}
    if not (bu.get("browser") is not None):
        missing.append("browser_use_tool.browser")
    if not (bu.get("llm") is not None):
        missing.append("browser_use_tool.llm")
    if missing:
        raise RuntimeError(
            "Missing required config fields: "
            + ", ".join(missing)
            + ". Please create config/browser_agent.yaml based on config/browser_agent.example.yaml"
        )

    server = BrowserAgentServer(config=config, prefix="", tags=["browser_agent"])
    server.register()

    # MCP server created from the app schema; we only include the browser_agent tag as tools.
    mcp_api = FastApiMCP(
        app,
        name="browser_agent_mcp",
        include_tags=["browser_agent"],
        headers=["authorization", "mcp-session-id"],
        session_resource_manager=server.session_resource_manager,
    )
    # Mount MCP endpoints onto the same APIRouter so routing stays prefixless.
    mcp_api.mount_sse(server.router, mount_path="/mcp")

    return app


def main():
    config_path = os.environ.get("BROWSER_AGENT_CONFIG", "config/browser_agent.yaml")
    config = _load_yaml_config(config_path)

    host = (config.get("server", {}) or {}).get("host", "0.0.0.0")
    port = int((config.get("server", {}) or {}).get("port", 9091))

    app = create_app(config)
    LOGGER.info("start browser_agent server at: http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


