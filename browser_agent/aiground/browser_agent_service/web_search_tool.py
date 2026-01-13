# -*- coding: utf-8 -*-
"""
web_search_tool

Web search tool adapter for OpenManus (open-source build).
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from aiground.framework.thirdparty.openmanus.app.config import config
from aiground.framework.thirdparty.openmanus.app.tool.base import BaseTool, ToolResult
from aiground.framework.thirdparty.openmanus.app.tool.search import (
    BaiduSearchEngine,
    BingSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleSearchEngine,
    WebSearchEngine,
)
from aiground.framework.thirdparty.openmanus.app.tool.search.base import SearchItem

LOGGER = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Represents a single search result returned by a search engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: int = Field(description="Position in search results")
    url: str = Field(description="URL of the search result")
    title: str = Field(default="", description="Title of the search result")
    description: str = Field(default="", description="Description or snippet of the search result")
    source: str = Field(description="The search engine that provided this result")
    raw_content: Optional[str] = Field(
        default=None, description="Raw content from the search result page if available"
    )

    def __str__(self) -> str:
        return f"{self.title} ({self.url})"


class SearchMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    total_results: int = Field(description="Total number of results found")
    language: str = Field(description="Language code used for the search")
    country: str = Field(description="Country code used for the search")


class SearchResponse(ToolResult):
    """Structured response from the web search tool, inheriting ToolResult."""

    query: str = Field(description="The search query that was executed")
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    metadata: Optional[SearchMetadata] = Field(default=None, description="Metadata about the search")

    @model_validator(mode="after")
    def populate_output(self) -> "SearchResponse":
        if self.error:
            return self

        result_text = [f"Search results for '{self.query}':"]
        for i, result in enumerate(self.results, 1):
            title = result.title.strip() or "No title"
            result_text.append(f"\n{i}. {title}")
            result_text.append(f"   URL: {result.url}")
            if getattr(result, "source", ""):
                result_text.append(f"   Source: {result.source}")
            if result.description.strip():
                result_text.append(f"   Description: {result.description}")
            if result.raw_content:
                content_preview = result.raw_content[:1000].replace("\n", " ").strip()
                if len(result.raw_content) > 1000:
                    content_preview += "..."
                result_text.append(f"   Content: {content_preview}")

        if self.metadata:
            result_text.extend(
                [
                    f"\nMetadata:",
                    f"- Total results: {self.metadata.total_results}",
                    f"- Language: {self.metadata.language}",
                    f"- Country: {self.metadata.country}",
                ]
            )
        self.output = "\n".join(result_text)
        return self


class WebContentFetcher:
    @staticmethod
    async def fetch_content(url: str, timeout: int = 10) -> Optional[str]:
        headers = {
            "WebSearch": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: requests.get(url, headers=headers, timeout=timeout)
            )
            if response.status_code != 200:
                LOGGER.warning("Failed to fetch content from %s: HTTP %s", url, response.status_code)
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            text = soup.get_text(separator="\n", strip=True)
            text = " ".join(text.split())
            return text[:10000] if text else None
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.warning("Error fetching content from %s: %s", url, e)
            return None


class WebSearchRequest(BaseModel):
    query: str = Field(description="(required) The search query to submit to the search engine.")
    num_results: int = Field(
        description="(optional) The number of search results to return. Default is 5.",
        default=5,
    )
    lang: str = Field(description="(optional) Language code for search results (default: en).", default="en")
    country: str = Field(description="(optional) Country code used for the search (default: us).", default="us")
    fetch_content: bool = Field(
        description="(optional) Whether to fetch full content from result pages. Default is false.",
        default=False,
    )


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = """Search the web for real-time information about any topic.
This tool returns comprehensive search results with relevant information, URLs, titles, and descriptions.
If the primary search engine fails, it automatically falls back to alternative engines."""

    _search_engine: dict[str, WebSearchEngine] = {
        # Open-source build: no internal TRAG engine; prefer Google (Serper) first.
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
        "bing": BingSearchEngine(),
    }
    content_fetcher: WebContentFetcher = WebContentFetcher()

    async def execute(
        self,
        query: str,
        num_results: int = 5,
        lang: Optional[str] = None,
        country: Optional[str] = None,
        fetch_content: bool = False,
    ) -> SearchResponse:
        retry_delay = getattr(config.search_config, "retry_delay", 60) if config.search_config else 60
        max_retries = getattr(config.search_config, "max_retries", 3) if config.search_config else 3
        if lang is None:
            lang = getattr(config.search_config, "lang", "en") if config.search_config else "en"
        if country is None:
            country = getattr(config.search_config, "country", "us") if config.search_config else "us"

        search_params = {"lang": lang, "country": country}

        for retry_count in range(max_retries + 1):
            results = await self._try_all_engines(query, num_results, search_params)
            if results:
                if fetch_content:
                    results = await self._fetch_content_for_results(results)
                return SearchResponse(
                    status="success",
                    query=query,
                    results=results,
                    metadata=SearchMetadata(
                        total_results=len(results),
                        language=lang,
                        country=country,
                    ),
                )

            if retry_count < max_retries:
                LOGGER.warning(
                    "All search engines failed. Waiting %s seconds before retry %s / %s...",
                    retry_delay,
                    retry_count + 1,
                    max_retries,
                )
                await asyncio.sleep(retry_delay)
            else:
                LOGGER.error("All search engines failed after %s retries. Giving up.", max_retries)

        return SearchResponse(
            query=query,
            error="All search engines failed to return results after multiple retries.",
            results=[],
        )

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[SearchResult]:
        engine_order = self._get_engine_order()
        failed_engines: List[str] = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            LOGGER.info("🔎 Attempting search with %s...", engine_name)
            try:
                search_items = await self._perform_search_with_engine(engine, query, num_results, search_params)
            except RetryError as re:
                last_exc = None
                try:
                    last_exc = re.last_attempt.exception()
                except Exception:
                    pass
                err_str = str(last_exc or re)
                LOGGER.warning("Search engine %s failed after retries: %s", engine_name, err_str)
                failed_engines.append(f"{engine_name}(error)")
                continue
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.warning("Search engine %s failed: %s", engine_name, e)
                failed_engines.append(f"{engine_name}(error)")
                continue

            if not search_items:
                failed_engines.append(f"{engine_name}(empty)")
                continue

            if failed_engines:
                LOGGER.info("Search successful with %s after trying: %s", engine_name, ", ".join(failed_engines))

            return [
                SearchResult(
                    position=i + 1,
                    url=item.url,
                    title=item.title or f"Result {i+1}",
                    description=item.description or "",
                    source=engine_name,
                )
                for i, item in enumerate(search_items)
            ]

        if failed_engines:
            LOGGER.error("All search engines failed: %s", ", ".join(failed_engines))
        return []

    async def _fetch_content_for_results(self, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return []
        tasks = [self._fetch_single_result_content(result) for result in results]
        fetched_results = await asyncio.gather(*tasks)
        return [
            (result if isinstance(result, SearchResult) else SearchResult(**result.dict()))
            for result in fetched_results
        ]

    async def _fetch_single_result_content(self, result: SearchResult) -> SearchResult:
        if result.url:
            content = await self.content_fetcher.fetch_content(result.url)
            if content:
                result.raw_content = content
        return result

    def _get_engine_order(self) -> List[str]:
        preferred = getattr(config.search_config, "engine", "").lower() if config.search_config else ""
        fallbacks = (
            [engine.lower() for engine in config.search_config.fallback_engines]
            if config.search_config and hasattr(config.search_config, "fallback_engines")
            else []
        )

        engine_order: List[str] = []
        for e in ("google", "duckduckgo", "bing", "baidu"):
            if e in self._search_engine and e not in engine_order:
                engine_order.append(e)
        if preferred and preferred in self._search_engine and preferred not in engine_order:
            engine_order.append(preferred)
        engine_order.extend([fb for fb in fallbacks if fb in self._search_engine and fb not in engine_order])
        engine_order.extend([e for e in self._search_engine if e not in engine_order])
        return engine_order

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: list(
                engine.perform_search(
                    query,
                    num_results=num_results,
                    lang=search_params.get("lang"),
                    country=search_params.get("country"),
                )
            ),
        )


def create_web_search_tool() -> WebSearchTool:
    return WebSearchTool()


