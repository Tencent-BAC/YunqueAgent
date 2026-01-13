# -*- coding: utf-8 -*-
"""
browser_use_tool

Browser-use tool (open-source build).
"""

import asyncio
import base64
import json
import logging
from typing import Literal, Optional

import markdownify
from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from aiground.common.dict_args import DictArgs
from aiground.framework.thirdparty.browser_use.browser.browser import (
    Browser as BrowserUseBrowser,
)
from aiground.framework.thirdparty.browser_use.browser.browser import (
    BrowserConfig,
    ProxySettings,
)
from aiground.framework.thirdparty.browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
)
from aiground.framework.thirdparty.browser_use.dom.service import DomService
from aiground.framework.thirdparty.mcp.server.session_resource import (
    SessionResource,
    SessionResourceManager,
)
from aiground.framework.thirdparty.openmanus.app.config import (
    BrowserSettings,
    LLMSettings,
    config,
)
from aiground.framework.thirdparty.openmanus.app.llm import LLM
from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult

from .web_search_tool import WebSearchTool

LOGGER = logging.getLogger(__name__)

# Trigger lazy-loading/animations before taking screenshots (avoid changing the final scroll position).
SCROLL_ANIMATION_PIXELS = 1000000
ANIMATION_SETTLE_SECONDS = 1

_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""


class BrowserUseRequest(BaseModel):
    action: Literal[
        "go_to_url",
        "click_element",
        "input_text",
        "scroll_down",
        "scroll_up",
        "scroll_to_text",
        "send_keys",
        "get_dropdown_options",
        "select_dropdown_option",
        "go_back",
        "web_search",
        "wait",
        "extract_content",
        "switch_tab",
        "open_tab",
        "close_tab",
    ] = Field(description="The browser action to perform")

    session_id: Optional[str] = Field(default=None, description="Session ID, optional")
    url: Optional[str] = Field(default=None, description="URL for 'go_to_url' or 'open_tab' actions")
    index: Optional[int] = Field(
        default=None,
        description=(
            "Element index for 'click_element', 'input_text',"
            " 'get_dropdown_options', or 'select_dropdown_option' actions"
        ),
    )
    text: Optional[str] = Field(
        default=None,
        description="Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
    )
    scroll_amount: Optional[int] = Field(
        default=None,
        description="Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
    )
    tab_id: Optional[int] = Field(default=None, description="Tab ID for 'switch_tab' action")
    query: Optional[str] = Field(default=None, description="Search query for 'web_search' action")
    goal: Optional[str] = Field(default=None, description="Extraction goal for 'extract_content' action")
    keys: Optional[str] = Field(default=None, description="Keys to send for keyboard actions")
    seconds: Optional[int] = Field(default=None, description="Seconds to wait for 'wait' action")

    @field_validator("*", mode="before")
    @classmethod
    def validate_parameters(cls, v, info: ValidationInfo) -> dict:
        action = info.data.get("action") if info.data else None
        if not action:
            return v
        dependencies = {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        }
        required_fields = dependencies.get(action, [])
        field_name = info.field_name
        if field_name in required_fields and v is None:
            raise ValueError(f"Field '{field_name}' is required when action is '{action}'")
        return v


class BrowserUseProperty(BaseModel):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    llm: Optional[LLM] = Field(default_factory=LLM)
    use_image: bool = Field(default=False, description="Whether to use image")
    image_resize_width: int = Field(default=1024, description="Image resize width")
    image_max_height: int = Field(default=10240, description="Image max height")
    keep_image_in_memory: bool = Field(default=True, description="Keep image in memory")
    keep_dom_in_memory: bool = Field(default=True, description="Keep DOM in memory")
    jina_api_key: str = Field(
        default="",
        description="Jina `r.jina.ai` API key token (without `Bearer ` prefix).",
    )
    viewport_expansion: int = Field(
        default=0,
        description="BrowserContext viewport_expansion (0 means current viewport only; -1 means full page DOM)",
    )

    class Config:
        arbitrary_types_allowed = True


class BrowserUseState(BaseModel, SessionResource):
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    async def destroy(self):
        await self.context.close()
        del self.dom_service


class BrowserUseStateSessionManager(SessionResourceManager):
    def __init__(self, browser: BrowserUseBrowser, viewport_expansion: int = 0):
        super().__init__()
        self._browser: BrowserUseBrowser = browser
        self._viewport_expansion = viewport_expansion
        self._session_resource = {}
        self._session_resource_lock = asyncio.Lock()

    async def create_temp_resource(self) -> BrowserUseState:
        return await self._create_session_state()

    async def create(self, session_id: str) -> None:
        if session_id in self._session_resource:
            return
        async with self._session_resource_lock:
            if session_id in self._session_resource:
                return
            self._session_resource[session_id] = await self._create_session_state()

    async def get_resource(self, session_id: str) -> Optional[BrowserUseState]:
        return self._session_resource.get(session_id, None)

    async def destroy(self, session_id: str) -> None:
        if session_id not in self._session_resource:
            return
        async with self._session_resource_lock:
            if session_id not in self._session_resource:
                return
            session_resource = self._session_resource.pop(session_id)
            try:
                await session_resource.destroy()
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.exception("destroy session resource error: %s", e)

    async def _create_session_state(self):
        context_config = BrowserContextConfig(
            highlight_elements=True,
            viewport_expansion=self._viewport_expansion,
        )
        if (
            config.browser_config
            and hasattr(config.browser_config, "new_context_config")
            and config.browser_config.new_context_config
        ):
            context_config = config.browser_config.new_context_config
            if hasattr(context_config, "viewport_expansion") and context_config.viewport_expansion is None:
                context_config.viewport_expansion = self._viewport_expansion

        context = await self._browser.new_context(context_config)
        dom_service = DomService(await context.get_current_page())
        return BrowserUseState(context=context, dom_service=dom_service)

    async def cleanup(self):
        self._browser = None
        for _, resource in self._session_resource.items():
            try:
                await resource.destroy()
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.error("destroy session resource error: %s", e)
        self._session_resource = {}


class BrowserUseTool:
    def __init__(self, property: BrowserUseProperty):
        self._property = property
        self._browser_store = BrowserUseStateSessionManager(
            property.browser, viewport_expansion=property.viewport_expansion
        )
        self._web_search_tool = WebSearchTool()
        self._action_fns = {
            "web_search": self._action_web_search,
            "go_to_url": self._action_go_to_url,
            "go_back": self._action_go_back,
            "click_element": self._action_click_element,
            "input_text": self._action_input_text,
            "scroll_down": self._action_scroll,
            "scroll_up": self._action_scroll,
            "scroll_to_text": self._action_scroll_to_text,
            "send_keys": self._action_send_keys,
            "get_dropdown_options": self._action_get_dropdown_options,
            "select_dropdown_option": self._action_select_dropdown_option,
            "extract_content": self._action_extract_content,
            "switch_tab": self._action_switch_tab,
            "open_tab": self._action_open_tab,
            "close_tab": self._action_close_tab,
            "wait": self._action_wait,
        }

    def get_llm(self) -> LLM:
        return self._property.llm

    async def create_session(self, session_id: str):
        await self._browser_store.create(session_id)

    async def destroy_session(self, session_id: str):
        await self._browser_store.destroy(session_id)

    async def _action_web_search(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        query = req.query
        if not query:
            return ToolResult(error="Query is required for 'web_search' action")
        search_response = await self._web_search_tool.execute(
            query=query, fetch_content=True, num_results=5
        )
        LOGGER.info("Web search results: %s", search_response)
        try:
            first_search_result = search_response.results[0] if search_response.results else None
            url_to_navigate = first_search_result.url if first_search_result else ""
        except Exception:  # pylint: disable=broad-except
            url_to_navigate = ""
        if url_to_navigate:
            try:
                page = await state.context.get_current_page()
                await page.goto(url_to_navigate)
                await page.wait_for_load_state()
            except Exception as e:  # pylint: disable=broad-except
                warn = (
                    "[Navigation warning] Failed to auto-navigate to first result: "
                    f"{url_to_navigate}\nReason: {e}"
                )
                LOGGER.warning(warn)
                try:
                    search_response.output = (search_response.output or "").rstrip() + "\n\n" + warn
                except Exception:
                    pass
        return search_response

    async def _action_go_to_url(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        url = req.url
        if not url:
            return ToolResult(error="URL is required for 'go_to_url' action")
        page = await state.context.get_current_page()
        await page.goto(url)
        await page.wait_for_load_state()
        return ToolResult(output=f"Navigated to {url}")

    async def _action_go_back(self, state: BrowserUseState, _: BrowserUseRequest) -> ToolResult:
        await state.context.go_back()
        return ToolResult(output="Navigated back")

    async def _action_click_element(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        index = req.index
        if index is None:
            return ToolResult(error="Index is required for 'click_element' action")
        try:
            tabs_before = await state.context.get_tabs_info()
        except Exception:  # pylint: disable=broad-except
            tabs_before = []

        element = await state.context.get_dom_element_by_index(index)
        if not element:
            return ToolResult(error=f"Element with index {index} not found")
        download_path = await state.context._click_element_node(element)
        output = f"Clicked element at index {index}"
        element_url = element.attributes.get("href", "") or element.attributes.get("src", "")
        if element_url and element_url.startswith("http"):
            output += f", Navigated to {element_url}"
        if download_path:
            output += f" - Downloaded file to {download_path}"

        try:
            tabs_after = await state.context.get_tabs_info()
            if len(tabs_after) > len(tabs_before):
                before_ids = {t.page_id for t in tabs_before}
                new_tabs = [t for t in tabs_after if t.page_id not in before_ids]
                new_tabs_non_blank = [t for t in new_tabs if t.url and t.url != "about:blank"]
                chosen = None
                if new_tabs_non_blank:
                    chosen = new_tabs_non_blank[-1]
                elif new_tabs:
                    chosen = new_tabs[-1]
                if chosen is not None:
                    await state.context.switch_to_tab(chosen.page_id)
                    output += f" - Switched to new tab [{chosen.page_id}]: {chosen.title} ({chosen.url})"
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.warning("auto switch to new tab failed: %s", e)
        return ToolResult(output=output)

    async def _action_input_text(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        index, text = req.index, req.text
        if index is None or not text:
            return ToolResult(error="Index and text are required for 'input_text' action")
        element = await state.context.get_dom_element_by_index(index)
        if not element:
            return ToolResult(error=f"Element with index {index} not found")
        await state.context._input_text_element_node(element, text)
        return ToolResult(output=f"Input '{text}' into element at index {index}")

    async def _action_scroll(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        action, scroll_amount = req.action, req.scroll_amount
        direction = 1 if action == "scroll_down" else -1
        amount = (
            scroll_amount
            if scroll_amount is not None
            else state.context.config.browser_window_size["height"]
        )
        await state.context.execute_javascript(f"window.scrollBy(0, {direction * amount});")
        return ToolResult(output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels")

    async def _action_scroll_to_text(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        text = req.text
        if not text:
            return ToolResult(error="Text is required for 'scroll_to_text' action")
        page = await state.context.get_current_page()
        try:
            locator = page.get_by_text(text, exact=False)
            await locator.scroll_into_view_if_needed()
            return ToolResult(output=f"Scrolled to text: '{text}'")
        except Exception as e:  # pylint: disable=broad-except
            return ToolResult(error=f"Failed to scroll to text: {str(e)}")

    async def _action_send_keys(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        keys = req.keys
        if not keys:
            return ToolResult(error="Keys are required for 'send_keys' action")
        page = await state.context.get_current_page()
        await page.keyboard.press(keys)
        return ToolResult(output=f"Sent keys: {keys}")

    async def _action_get_dropdown_options(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        index = req.index
        if index is None:
            return ToolResult(error="Index is required for 'get_dropdown_options' action")
        element = await state.context.get_dom_element_by_index(index)
        if not element:
            return ToolResult(error=f"Element with index {index} not found")
        page = await state.context.get_current_page()
        options = await page.evaluate(
            """
            (xpath) => {
                const select = document.evaluate(xpath, document, null,
                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                if (!select) return null;
                return Array.from(select.options).map(opt => ({
                    text: opt.text,
                    value: opt.value,
                    index: opt.index
                }));
            }
            """,
            element.xpath,
        )
        return ToolResult(output=f"Dropdown options: {options}")

    async def _action_select_dropdown_option(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        index, text = req.index, req.text
        if index is None or not text:
            return ToolResult(error="Index and text are required for 'select_dropdown_option' action")
        element = await state.context.get_dom_element_by_index(index)
        if not element:
            return ToolResult(error=f"Element with index {index} not found")
        page = await state.context.get_current_page()
        await page.select_option(element.xpath, label=text)
        return ToolResult(output=f"Selected option '{text}' from dropdown at index {index}")

    async def _action_extract_content(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        goal = req.goal
        if not goal:
            return ToolResult(error="Goal is required for 'extract_content' action")
        page = await state.context.get_current_page()
        content = markdownify.markdownify(await page.content())

        max_content_length = getattr(config.browser_config, "max_content_length", 2000)
        prompt = f"""\
Your task is to extract the content of the page. \
You will be given a page and a goal, and you should extract all relevant information around this goal from the page. \
If the goal is vague, summarize the page. Respond in json format.
Extraction goal: {goal}
"""
        user_message = f"""
Page content:
{content[:max_content_length]}
"""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ]

        extraction_function = {
            "type": "function",
            "function": {
                "name": "extract_content",
                "description": "Extract specific information from a webpage based on a goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "extracted_content": {
                            "type": "object",
                            "description": "The content extracted from the page according to the goal",
                            "properties": {
                                "text": {"type": "string", "description": "Text content extracted from the page"},
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional metadata about the extracted content",
                                    "properties": {
                                        "source": {"type": "string", "description": "Source of the extracted content"}
                                    },
                                },
                            },
                        }
                    },
                    "required": ["extracted_content"],
                },
            },
        }

        response = await self._property.llm.ask_tool(
            messages,
            tools=[extraction_function],
            tool_choice="required",
        )
        if response and response.tool_calls:
            args = json.loads(response.tool_calls[0].function.arguments)
            extracted_content = args.get("extracted_content", {})
            return ToolResult(output=f"Extracted from page:\n{extracted_content}\n")
        return ToolResult(output="No content was extracted from the page.")

    async def _action_switch_tab(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        tab_id = req.tab_id
        if tab_id is None:
            return ToolResult(error="Tab ID is required for 'switch_tab' action")
        await state.context.switch_to_tab(tab_id)
        page = await state.context.get_current_page()
        await page.wait_for_load_state()
        return ToolResult(output=f"Switched to tab {tab_id}")

    async def _action_open_tab(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        url = req.url
        if not url:
            return ToolResult(error="URL is required for 'open_tab' action")
        await state.context.create_new_tab(url)
        return ToolResult(output=f"Opened new tab with {url}")

    async def _action_close_tab(self, state: BrowserUseState, _: BrowserUseRequest) -> ToolResult:
        await state.context.close_current_tab()
        return ToolResult(output="Closed current tab")

    async def _action_wait(self, state: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        seconds_to_wait = req.seconds if req.seconds is not None else 3
        await asyncio.sleep(seconds_to_wait)
        return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

    async def _action_unknown(self, _: BrowserUseState, req: BrowserUseRequest) -> ToolResult:
        return ToolResult(error=f"Unknown action: {req.action}")

    async def execute(self, req: BrowserUseRequest, session_id: str) -> ToolResult:
        state: BrowserUseState = None
        try:
            if not session_id:
                state = await self._browser_store.create_temp_resource()
            else:
                state = await self._browser_store.get_resource(session_id)
            if not state:
                return ToolResult(error=f"Session not found: {session_id}")
            action_fn = self._action_fns.get(req.action, self._action_unknown)
            result = await action_fn(state, req)
            return result
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Error executing browser action '%s'", req.action)
            return ToolResult(error=f"Browser action '{req.action}' failed: {str(e)}")
        finally:
            if not session_id and state is not None:
                await state.destroy()

    async def get_current_state(self, session_id: str) -> ToolResult:
        try:
            browser_state: BrowserUseState = await self._browser_store.get_resource(session_id)
            ctx = browser_state.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state(cache_clickable_elements_hashes=False)
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            page = await ctx.get_current_page()
            await page.bring_to_front()
            await page.wait_for_load_state()

            try:
                initial_scroll_y = await ctx.execute_javascript("window.scrollY;")
                bottom_y = await ctx.execute_javascript(
                    "Math.max(0, document.documentElement.scrollHeight - window.innerHeight);"
                )
                await ctx.execute_javascript(f"window.scrollTo(0, {bottom_y});")
                await asyncio.sleep(ANIMATION_SETTLE_SECONDS)
                await ctx.execute_javascript(f"window.scrollTo(0, {initial_scroll_y});")
                await asyncio.sleep(ANIMATION_SETTLE_SECONDS / 2)
            except Exception as e:  # pylint: disable=broad-except
                LOGGER.warning("pre-screenshot scroll failed: %s", e)

            clickable_elements = (
                state.element_tree.clickable_elements_to_string(include_attributes=["href", "src"])
                if state.element_tree
                else ""
            )
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": (
                    "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. "
                    "Clicking on these indices will navigate to or interact with the respective content behind them."
                ),
                "interactive_elements": clickable_elements,
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
                "web_data": await page.content(),
            }

            await ctx.remove_highlights()
            remove_screenshot = await page.screenshot(full_page=True, animations="disabled", type="jpeg", quality=100)
            remove_screenshot = base64.b64encode(remove_screenshot).decode("utf-8")
            return ToolResult(output=json.dumps(state_info, indent=4, ensure_ascii=False), base64_image=remove_screenshot)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Failed to get browser state: %s", e)
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def get_current_content(self, session_id: str) -> ToolResult:
        try:
            browser_state: BrowserUseState = await self._browser_store.get_resource(session_id)
            ctx = browser_state.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")
            tabs = await ctx.get_tabs_info()
            extra_tabs = [tab.model_dump() for tab in tabs] if tabs else []
            page_state = await ctx.get_state(cache_clickable_elements_hashes=True)
            state_info = {
                "content": (
                    page_state.element_tree.clickable_elements_to_string(include_attributes=["href", "src"])
                    if page_state.element_tree
                    else "{}"
                ),
                "extra_tabs": extra_tabs,
            }
            return ToolResult(output=json.dumps(state_info, ensure_ascii=False))
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception("Failed to get browser state: %s", e)
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def _cleanup(self):
        await self._browser_store.cleanup()

    def __del__(self):
        if self._property.browser is not None:
            try:
                asyncio.run(self._cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._cleanup())
                loop.close()


def create_browser_use_tool(config_args: DictArgs) -> BrowserUseTool:
    def create_llm(llm_args: DictArgs):
        final_llm_config = {}
        for k, v in llm_args.items():
            if not v or not isinstance(v, dict):
                continue
            final_llm_config[k] = LLMSettings.model_validate(v, strict=False)
        return LLM(llm_args.name, final_llm_config)

    def create_browser_use(browser_args: DictArgs):
        browser_config = BrowserSettings.model_validate(browser_args, strict=False)
        browser_config_kwargs = {"headless": False, "disable_security": True}
        if browser_config.proxy and browser_config.proxy.server:
            browser_config_kwargs["proxy"] = ProxySettings(
                server=browser_config.proxy.server,
                username=browser_config.proxy.username,
                password=browser_config.proxy.password,
            )

        browser_attrs = [
            "headless",
            "disable_security",
            "extra_chromium_args",
            "chrome_instance_path",
            "extra_browser_args",
            "browser_binary_path",
            "chrome_remote_debugging_port",
            "wss_url",
            "cdp_url",
        ]
        for attr in browser_attrs:
            value = getattr(browser_config, attr, None)
            if value is not None:
                if not isinstance(value, list) or value:
                    browser_config_kwargs[attr] = value

        return BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

    browser_use_property = BrowserUseProperty(
        browser=create_browser_use(config_args.browser),
        llm=create_llm(config_args.llm),
        use_image=config_args.get("use_image", False),
        image_resize_width=config_args.get("image_resize_width", 1024),
        image_max_height=config_args.get("image_max_height", 10240),
        keep_image_in_memory=config_args.get("keep_image_in_memory", True),
        keep_dom_in_memory=config_args.get("keep_dom_in_memory", True),
        jina_api_key=config_args.get("jina_api_key", ""),
        viewport_expansion=config_args.get("viewport_expansion", 0),
    )
    return BrowserUseTool(browser_use_property)


