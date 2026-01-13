# -*- coding: utf-8 -*-
"""
url_to_markdown_tool

Convert an arbitrary URL to markdown using Jina AI's `r.jina.ai/` endpoint.

Example:
  https://arxiv.org/pdf/2501.14249
  -> https://r.jina.ai/arxiv.org/pdf/2501.14249
"""

import logging
from typing import Tuple
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult

LOGGER = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # allow passing "arxiv.org/pdf/..." without scheme
    return "https://" + u.lstrip("/")


def _to_jina_url(url: str) -> str:
    """
    Convert an arbitrary URL to Jina Reader URL.

    We prefer the host/path form:
      https://r.jina.ai/<host><path>?<query>

    Example:
      https://www.baidu.com/ -> https://r.jina.ai/www.baidu.com/

    If the URL is already a r.jina.ai URL, return as-is.
    """
    url = _normalize_url(url)
    parsed = urlparse(url)
    if parsed.netloc == "r.jina.ai":
        return url
    if not parsed.netloc:
        # Fallback: keep the original behavior for unexpected inputs.
        return "https://r.jina.ai/" + url
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"https://r.jina.ai/{parsed.netloc}{path}{query}"


def _fetch_sync(url: str, timeout: int, bearer_token: str = "") -> Tuple[str, int, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/markdown,text/plain,*/*",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.text, resp.status_code, resp.headers.get("content-type", "") or ""


class UrlToMarkdownRequest(BaseModel):
    url: str = Field(description="The original URL to convert to markdown.")
    start: int = Field(
        default=0,
        description="Start offset (0-based) into the converted markdown for paging.",
    )
    max_chars: int = Field(
        default=20000,
        description="Page size: maximum characters to return from `start`.",
    )
    timeout: int = Field(default=20, description="HTTP timeout in seconds.")


class UrlToMarkdownTool:
    def __init__(self, bearer_token: str = ""):
        # Jina API key token (without `Bearer ` prefix). Read from config/env, never hardcode.
        self._bearer_token = (bearer_token or "").strip()

    async def execute(self, req: UrlToMarkdownRequest, _: str) -> ToolResult:
        try:
            src_url = (req.url or "").strip()
            if not src_url:
                return ToolResult(error="url_to_markdown: url is required")

            jina_url = _to_jina_url(src_url)
            md, status, content_type = _fetch_sync(
                jina_url, timeout=req.timeout, bearer_token=self._bearer_token
            )

            total_chars = len(md or "")
            start = int(req.start or 0)
            if start < 0:
                start = 0
            if start > total_chars:
                start = total_chars
            page_size = int(req.max_chars or 20000)
            if page_size <= 0:
                page_size = 20000
            end = min(total_chars, start + page_size)
            has_more = end < total_chars
            next_start = end if has_more else None

            page = md[start:end]

            meta = (
                "[url_to_markdown]\n"
                f"source_url: {src_url}\n"
                f"jina_url: {jina_url}\n"
                f"status: {status}\n"
                f"content_type: {content_type}\n\n"
                f"page_start: {start}\n"
                f"page_end: {end}\n"
                f"total_chars: {total_chars}\n"
                f"has_more: {has_more}\n"
                f"next_start: {next_start}\n\n"
            )
            return ToolResult(output=meta + page)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("url_to_markdown failed: %s", e)
            return ToolResult(error=f"url_to_markdown failed: {e}")


def create_url_to_markdown_tool(bearer_token: str = "") -> UrlToMarkdownTool:
    return UrlToMarkdownTool(bearer_token=bearer_token)


