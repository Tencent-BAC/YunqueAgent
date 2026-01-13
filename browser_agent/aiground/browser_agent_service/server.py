# -*- coding: utf-8 -*-
"""
browser_agent_service.server

Open-source browser agent service: prefixless HTTP APIs, mountable via FastApiMCP at `/mcp`.
"""

import asyncio
import logging
import os
import uuid
from typing import List, Literal, Optional

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from aiground.common.dict_args import DictArgs
from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult

from .browser_use_agent import BrowserUseAgent, BrowserUseTaskRequest
from .browser_use_tool import BrowserUseRequest, create_browser_use_tool
from .url_to_markdown_tool import UrlToMarkdownRequest, create_url_to_markdown_tool
from .web_search_tool import SearchResponse, WebSearchRequest, create_web_search_tool

LOGGER = logging.getLogger(__name__)
MCP_SESSION_ID_HEADER = "mcp-session-id"


class PingRequest(BaseModel):
    ping: Literal["ping"] = Field(description="ping")


class BrowserUseSessionRequest(BaseModel):
    session_id: Optional[str] = Field(description="session id if provided", default=None)


class BrowserAgentServer:
    """
    Pure FastAPI router-based browser agent service.

    - `router` can be included into a FastAPI app
    - `session_resource_manager` is passed to FastApiMCP for MCP session resource management
    """

    def __init__(self, config: DictArgs, prefix: str = "", tags: Optional[List[str]] = None):
        self._config = config
        self.router = APIRouter(prefix=prefix or "", tags=tags or ["browser_agent"])

        # Compatibility: allow injecting Serper key from YAML into env var SERPER_KEY_ID.
        web_search_cfg = config.get("web_search_tool", DictArgs({}))
        serper_key = web_search_cfg.get("serper_key_id", "") or web_search_cfg.get(
            "SERPER_KEY_ID", ""
        )
        if serper_key:
            os.environ["SERPER_KEY_ID"] = str(serper_key)

        trace_config = config.get("trace", DictArgs({}))
        trace_enabled = bool(
            trace_config.get("enabled", False) or trace_config.get("enable", False)
        )
        trace_data_dir = (trace_config.get("data_dir", "") or "").strip() if trace_enabled else ""
        self._browser_use_tool = create_browser_use_tool(config.browser_use_tool)
        self._browser_use_agent = BrowserUseAgent(
            self._browser_use_tool, trace_data_dir
        )
        self._web_search_tool = create_web_search_tool()
        self._url_to_markdown_tool = create_url_to_markdown_tool(
            bearer_token=getattr(self._browser_use_tool._property, "jina_api_key", "") or ""
        )
        self.session_resource_manager = self._browser_use_tool._browser_store

    def register(self) -> None:
        self.router.add_api_route("/ping", self.ping, operation_id="ping", methods=["POST"])
        self.router.add_api_route(
            "/browser_use",
            self.browser_use,
            operation_id="browser_use",
            methods=["POST"],
        )
        self.router.add_api_route(
            "/web_search",
            self.web_search,
            operation_id="web_search",
            response_model=SearchResponse,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/url_to_markdown",
            self.url_to_markdown,
            operation_id="url_to_markdown",
            methods=["POST"],
        )
        self.router.add_api_route(
            "/create_browser_session",
            self.create_browser_use_session,
            operation_id="create_browser_use_session",
            methods=["POST"],
        )
        self.router.add_api_route(
            "/destroy_browser_session",
            self.destroy_browser_use_session,
            operation_id="destroy_browser_use_session",
            methods=["POST"],
        )
        self.router.add_api_route(
            "/get_current_browser_state",
            self.browser_use_state,
            operation_id="get_current_browser_state",
            methods=["GET"],
        )
        self.router.add_api_route(
            "/browser_use_agent",
            self.browser_use_agent,
            operation_id="browser_use_agent",
            methods=["POST"],
        )

    async def ping(self, ping: PingRequest, request: Request) -> str:  # noqa: ARG002
        session_id = request.headers.get(MCP_SESSION_ID_HEADER)
        return f"pong: {session_id}"

    async def create_browser_use_session(self, req: BrowserUseSessionRequest) -> ToolResult:
        session_id = req.session_id or uuid.uuid4().hex
        try:
            await self.session_resource_manager.create(session_id)
            return ToolResult(output=f"{session_id}")
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("create session resource error: %s", e)
            return ToolResult(error=f"session create error: {e}")

    async def destroy_browser_use_session(self, req: BrowserUseSessionRequest) -> ToolResult:
        session_id = req.session_id
        if not session_id:
            return ToolResult(error="session id is required")
        try:
            await self.session_resource_manager.destroy(session_id)
            return ToolResult(output=f"session destroyed: {session_id}")
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("destroy session resource error: %s", e)
            return ToolResult(error=f"session destroy error: {e}")

    async def web_search(self, req: WebSearchRequest) -> SearchResponse:
        return await self._web_search_tool.execute(
            req.query, req.num_results, req.lang, req.country, req.fetch_content
        )

    async def url_to_markdown(self, req: UrlToMarkdownRequest) -> ToolResult:
        return await self._url_to_markdown_tool.execute(req, "")

    async def browser_use(self, req: BrowserUseRequest, request: Request) -> ToolResult:
        session_id = request.headers.get(MCP_SESSION_ID_HEADER, req.session_id)
        if not session_id:
            return ToolResult(error="session id is required")
        return await self._browser_use_tool.execute(req, session_id)

    async def browser_use_state(
        self,
        req: BrowserUseSessionRequest = Query(...),
        mcp_session_id: str = Header(alias=MCP_SESSION_ID_HEADER, default=""),
    ) -> ToolResult:
        session_id = mcp_session_id or req.session_id
        if not session_id:
            return ToolResult(error="session id is required")
        return await self._browser_use_tool.get_current_state(session_id)

    async def browser_use_agent(self, req: BrowserUseTaskRequest):
        generator_fn, cleanup_fn = await self._browser_use_agent.execute(req)
        return StreamingResponse(
            generator_fn,
            media_type="text/event-stream",
            background=BackgroundTask(cleanup_fn),
        )


def run_import_smoke_check() -> None:
    """
    Smoke test helper (local/CI): verify the module can be imported and `register()` works.
    """
    cfg = DictArgs({"browser_use_tool": {}, "web_search_tool": {}, "trace": {}})
    s = BrowserAgentServer(cfg, prefix="", tags=["browser_agent"])
    s.register()


if __name__ == "__main__":  # pragma: no cover
    # Simple self-check (does not start a server).
    asyncio.run(asyncio.to_thread(run_import_smoke_check))


