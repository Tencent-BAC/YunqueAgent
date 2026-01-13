from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from aiground.framework.thirdparty.browser_use.dom.history_tree_processor.service import (
    DOMHistoryElement,
)
from aiground.framework.thirdparty.browser_use.dom.views import DOMState


# Pydantic
class TabInfo(BaseModel):
    """Represents information about a browser tab"""

    page_id: int
    url: str
    title: str
    parent_page_id: Optional[int] = (
        None  # parent page that contains this popup or cross-origin iframe
    )


@dataclass
class BrowserState(DOMState):
    url: str
    title: str
    tabs: List[TabInfo]
    screenshot: Optional[str] = None
    pixels_above: int = 0
    pixels_below: int = 0
    browser_errors: List[str] = field(default_factory=list)


@dataclass
class BrowserStateHistory:
    url: str
    title: str
    tabs: List[TabInfo]
    interacted_element: Union[List[Optional[DOMHistoryElement]], List[None]]
    screenshot: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {}
        data["tabs"] = [tab.model_dump() for tab in self.tabs]
        data["screenshot"] = self.screenshot
        data["interacted_element"] = [
            el.to_dict() if el else None for el in self.interacted_element
        ]
        data["url"] = self.url
        data["title"] = self.title
        return data


class BrowserError(Exception):
    """Base class for all browser errors"""


class URLNotAllowedError(BrowserError):
    """Error raised when a URL is not allowed"""
