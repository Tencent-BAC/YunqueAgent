import json
import os
from typing import Any, Dict, List, Optional

import http.client

try:
    from googlesearch import search as googlesearch_search  # type: ignore
except Exception:  # pragma: no cover
    googlesearch_search = None  # type: ignore

from aiground.framework.thirdparty.openmanus.app.tool.search.base import SearchItem, WebSearchEngine


def _contains_chinese_basic(text: str) -> bool:
    return any("\u4E00" <= ch <= "\u9FFF" for ch in text)


def _norm_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    return str(lang).strip().lower()


def _norm_country(country: Optional[str]) -> str:
    if not country:
        return "us"
    return str(country).strip().lower()


def _serper_location_for(country: str, lang: str, query: str) -> str:
    """
    Serper accepts a free-text `location`. Use a conservative default to avoid surprising geo bias.
    """
    if country == "cn" or _contains_chinese_basic(query) or lang.startswith("zh"):
        return "China"
    if country == "us":
        return "United States"
    # Fallback: let Serper decide if location is not provided.
    return ""


class GoogleSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        # Prefer Serper API (stable + supports server-side execution). Fallback to `googlesearch` if no key.
        serper_key = os.environ.get("SERPER_KEY_ID", "").strip()
        lang = _norm_lang(kwargs.get("lang"))
        country = _norm_country(kwargs.get("country"))

        if serper_key:
            location = _serper_location_for(country=country, lang=lang, query=query)
            body: Dict[str, Any] = {"q": query, "num": int(num_results), "gl": country, "hl": lang}
            if location:
                body["location"] = location

            conn = http.client.HTTPSConnection("google.serper.dev", timeout=10)
            payload = json.dumps(body, ensure_ascii=False)
            conn.request(
                "POST",
                "/search",
                body=payload.encode("utf-8"),
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
            )
            res = conn.getresponse()
            raw = res.read()
            if res.status < 200 or res.status >= 300:
                raise RuntimeError(
                    f"Serper request failed (HTTP {res.status}): {raw[:200].decode('utf-8', errors='ignore')}"
                )
            data = json.loads(raw.decode("utf-8"))
            organic = data.get("organic") or []

            results: List[SearchItem] = []
            for item in organic[: int(num_results)]:
                title = (item.get("title") or "").strip()
                url = (item.get("link") or item.get("url") or "").strip()
                desc = (item.get("snippet") or item.get("description") or "").strip()
                if not url:
                    continue
                results.append(SearchItem(title=title or "No title", url=url, description=desc))
            return results

        # Fallback: old behavior (may be rate-limited / blocked in server environments)
        if googlesearch_search is None:
            raise RuntimeError("Google search is not configured: SERPER_KEY_ID missing and googlesearch unavailable.")

        raw_results = googlesearch_search(query, num_results=num_results, advanced=True)
        results: List[SearchItem] = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                results.append(SearchItem(title=f"Google Result {i+1}", url=item, description=""))
            else:
                results.append(
                    SearchItem(
                        title=getattr(item, "title", "") or f"Google Result {i+1}",
                        url=getattr(item, "url", ""),
                        description=getattr(item, "description", "") or "",
                    )
                )
        return results
