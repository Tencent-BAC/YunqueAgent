# -*- coding: utf-8 -*-
"""
browser_use_agent

Browser-use agent (open-source build).
"""

import asyncio
import base64
import copy
import datetime
import io
import json
import logging
import os
import typing
import uuid
from abc import ABC
from typing import Any, List, Optional

from PIL import Image
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from pydantic import BaseModel, Field

from aiground.framework.thirdparty.openmanus.app.agent.toolcall import (
    TOOL_CALL_REQUIRED,
)
from aiground.framework.thirdparty.openmanus.app.exceptions import TokenLimitExceeded
from aiground.framework.thirdparty.openmanus.app.schema import (
    ROLE_TYPE,
    Memory,
    Message,
    ToolCall,
    ToolChoice,
)
from aiground.framework.thirdparty.openmanus.app.tool.base import ToolResult
from aiground.framework.thirdparty.openmanus.app.tracer import Tracer

from .browser_use_tool import _BROWSER_DESCRIPTION, BrowserUseRequest, BrowserUseTool
from .url_to_markdown_tool import UrlToMarkdownRequest, UrlToMarkdownTool

LOGGER = logging.getLogger(__name__)


class BrowserUseTaskRequest(BaseModel):
    """BrowserUseAgentRequest"""

    task: str = Field(description="task")
    max_steps: int = Field(description="max step num", default=10)


class BaseToolFunction(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    request_class: typing.Type[BaseModel] = None
    tool_func: typing.Callable[[BaseModel, str], typing.Coroutine[Any, Any, ToolResult]]

    class Config:
        arbitrary_types_allowed = True

    def to_param(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, args: dict, session_id: str) -> ToolResult:
        req = self.request_class.model_validate(args)
        return await self.tool_func(req, session_id)


class ToolFunctionList(BaseModel):
    tools: List[BaseToolFunction] = Field(description="tool function list")

    def to_param(self) -> List[dict]:
        return [tool.to_param() for tool in self.tools]

    def has(self, name: str):
        for tool in self.tools:
            if tool.name == name:
                return True
        return False

    async def execute(self, name: str, args: dict, session_id: str) -> ToolResult:
        for tool in self.tools:
            if tool.name == name:
                return await tool.call(args, session_id)
        return ToolResult(output=f"Tool {name} not found")


_TERMINATE_DESCRIPTION = (
    "Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task."
    "When you have finished all the tasks, call this tool to end the work."
)


class TerminateToolFunction(BaseToolFunction):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }


class TerminateRequest(BaseModel):
    status: str = Field(
        description="The finish status of the interaction.",
        examples=["success", "failure"],
    )


class TerminateTool:
    async def execute(self, status: str, _: str) -> ToolResult:
        return ToolResult(output=f"The interaction has been completed with status: {status}")


_URL_TO_MARKDOWN_DESCRIPTION = (
    "Convert a URL to markdown by calling `https://r.jina.ai/<host><path>` and return the markdown text. "
    "Use this when you need to extract text from a document URL (e.g. arXiv PDF) quickly."
)


class UrlToMarkdownToolFunction(BaseToolFunction):
    name: str = "url_to_markdown"
    description: str = _URL_TO_MARKDOWN_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The original URL to convert."},
            "start": {
                "type": "integer",
                "description": "Start offset (0-based) into the converted markdown for paging (default 0).",
            },
            "max_chars": {
                "type": "integer",
                "description": "Page size: maximum characters to return from `start` (default 20000).",
            },
            "timeout": {"type": "integer", "description": "HTTP timeout in seconds (default 20)."},
        },
        "required": ["url"],
    }


class BrowserUseToolFunction(BaseToolFunction):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
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
                ],
                "description": "The browser action to perform",
            },
            "url": {"type": "string", "description": "URL for 'go_to_url' or 'open_tab' actions"},
            "index": {
                "type": "integer",
                "description": (
                    "Element index for 'click_element', 'input_text', 'get_dropdown_options', "
                    "or 'select_dropdown_option' actions"
                ),
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": (
                    "Pixels to scroll (positive for down, negative for up) "
                    "for 'scroll_down' or 'scroll_up' actions"
                ),
            },
            "tab_id": {"type": "integer", "description": "Tab ID for 'switch_tab' action"},
            "query": {"type": "string", "description": "Search query for 'web_search' action"},
            "goal": {"type": "string", "description": "Extraction goal for 'extract_content' action"},
            "keys": {"type": "string", "description": "Keys to send for 'send_keys' action"},
            "seconds": {"type": "integer", "description": "Seconds to wait for 'wait' action"},
        },
        "required": ["action"],
    }


TASK_EXECUTOR_SYSTEM_PROMPT = (
    "You are AIMate, an all-capable AI assistant, aimed at solving any task presented by the user. "
    "You have various tools at your disposal that you can call upon to efficiently complete complex requests. "
    "Whether it's information retrieval or web browsing you can handle it all."
    "\n\nTOOL CALL LIMIT: In each assistant turn, you may call AT MOST ONE tool/function. "
    "Do not request multiple tool calls in a single response. If you need multiple actions, do them across multiple turns."
    "\n\nIMPORTANT PDF RULE: If the user provides a PDF link (e.g., an arXiv /pdf/ URL), DO NOT try to open/read the PDF in the browser. "
    "Instead, call the `url_to_markdown` tool first to convert the PDF/page into markdown text, then read/quote from that extracted text."
    "\n\nEPHEMERAL CONTEXT RULE (VERY IMPORTANT): Screenshots and `url_to_markdown` outputs are ONLY AVAILABLE FOR THE CURRENT TURN and will NOT be kept in later turns. "
    "Therefore, right after you call `url_to_markdown` OR right after you inspect a screenshot/DOM and find important facts (numbers, dates, names, quotes, key claims), "
    "you MUST immediately write a short plain-text summary in the assistant message BEFORE moving on. "
    "If you used `url_to_markdown` in the previous step/turn, start your next assistant message by summarizing the extracted content first."
    "If you want to stop interaction, use `terminate` tool/function call."
    "If a pop-up or a CAPTCHA appears on the page, "
    "you need to close the pop-up and complete the CAPTCHA.\n"
    "DO NOT ASSUME OR INFER UNVERIFIED DETAILS."
)


BROWSER_NEXT_STEP_PROMPT = """
[Current Browser State]
--------------------------------------------------
**URL:** {url_placeholder}
**Title:** {title_placeholder}
**Tabs:** {tabs_placeholder}

**Interactive Elements:**
{elements_placeholder}
--------------------------------------------------

**Last Action Results:**
{results_placeholder}

IMPORTANT NOTE:
- Screenshots and `url_to_markdown` outputs are only available for the current turn and will NOT be kept in later turns.
- If the previous step used `url_to_markdown` or relied on a screenshot, summarize the key facts you extracted in your next assistant message.
"""


class ReasoningResult(BaseModel):
    should_act: bool = Field(default=False, description="Whether the agent should act")
    tool_calls: Optional[List[ChatCompletionMessageToolCallUnion]] = Field(
        default=None, description="Tool calls"
    )
    content: str = Field(description="Content", default="")


class BrowserUseAgent:
    def __init__(self, tool: BrowserUseTool, trace_data_dir: Optional[str] = None):
        self._tool = tool
        self._trace_data_dir = trace_data_dir

    def init_asyncio_loop(self):
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    def async_run(self, task):
        self.init_asyncio_loop()
        asyncio.run(task)

    async def execute(self, request: BrowserUseTaskRequest):
        task_executor = BrowserUseTaskExecutor(request.task, self._tool, self._trace_data_dir)
        await task_executor.initialize_session()

        async def generator():
            step_cnt = 0
            max_turns = request.max_steps
            while True:
                if step_cnt >= max_turns:
                    msg = await task_executor.final_execute()
                    yield "data: " + json.dumps({"code": 201, "message": "success", "data": msg}, ensure_ascii=False) + "\n\n"
                    break
                if task_executor.is_finished():
                    msg = task_executor.final_result()
                    yield "data: " + json.dumps({"code": 200, "message": "success", "data": msg}, ensure_ascii=False) + "\n\n"
                    break
                step_cnt += 1
                task = asyncio.create_task(task_executor.execute())
                while True:
                    msg = await task_executor._message_queue.get()
                    if msg == "STEP_DONE":
                        break
                    yield "data: " + json.dumps({"code": 100, "message": "success", "data": msg}, ensure_ascii=False) + "\n\n"
                await task

        async def cleanup():
            await task_executor.cleanup_session()

        return generator(), cleanup


class BrowserUseTaskExecutor:
    def __init__(self, question: str, tool: BrowserUseTool, trace_data_dir: Optional[str] = None):
        self.name = "AIMate"
        self._question = question
        self._tool = tool
        self._llm = self._tool.get_llm()
        self.memory = Memory()
        self._session_id = uuid.uuid4().hex
        self._trace_data_dir = trace_data_dir
        self._available_tools = ToolFunctionList(
            tools=[
                BrowserUseToolFunction(request_class=BrowserUseRequest, tool_func=self._tool.execute),
                UrlToMarkdownToolFunction(request_class=UrlToMarkdownRequest, tool_func=self._url_to_markdown_execute),
                TerminateToolFunction(request_class=TerminateRequest, tool_func=TerminateTool().execute),
            ]
        )
        self._tool_choice = ToolChoice.AUTO
        self._current_system_prompt = None
        self._current_step = 0
        self._finish_reason = "unknown"
        self._last_action_results: str = ""
        self._update_memory("user", self._question)
        self._tracer: Optional[Tracer] = None
        self._trace_request_dir: Optional[str] = None
        self._trace_file_path: Optional[str] = None
        self._trace_images_dir: Optional[str] = None

    def _resolve_trace_dirs(self) -> None:
        self._trace_request_dir = None
        self._trace_file_path = None
        self._trace_images_dir = None

        root = (self._trace_data_dir or "").strip()
        if not root:
            return
        self._trace_request_dir = os.path.join(root, self._session_id)
        self._trace_file_path = os.path.join(self._trace_request_dir, "trace.jsonl")
        self._trace_images_dir = os.path.join(self._trace_request_dir, "images")

    async def _url_to_markdown_execute(self, req: UrlToMarkdownRequest, _: str) -> ToolResult:
        token = getattr(self._tool._property, "jina_api_key", "") or ""
        return await UrlToMarkdownTool(bearer_token=token).execute(req, "")

    def _process_image(self, b64_img: str) -> str:
        if not b64_img:
            return b64_img
        try:
            target_width = getattr(self._tool._property, "image_resize_width", 1024)
            max_height = getattr(self._tool._property, "image_max_height", 10240)
            img_data = base64.b64decode(b64_img)
            img = Image.open(io.BytesIO(img_data))
            w, h = img.size
            if w and target_width and w != target_width:
                ratio = float(target_width) / float(w)
                new_h = int(h * ratio)
                img = img.resize((int(target_width), int(new_h)))
            else:
                new_h = h
            if max_height and new_h > max_height:
                img = img.crop((0, 0, int(target_width), int(max_height)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error("Image processing failed: %s", e)
            return b64_img

    def _clean_memory(self):
        if not self.memory or not self.memory.messages:
            return
        messages = self.memory.messages
        keep_image = getattr(self._tool._property, "keep_image_in_memory", True)
        keep_dom = getattr(self._tool._property, "keep_dom_in_memory", True)

        last_assistant_idx = -1
        for i, m in enumerate(messages):
            if getattr(m, "role", "") == "assistant":
                last_assistant_idx = i

        for i, msg in enumerate(messages[:-1]):
            if not keep_image and getattr(msg, "base64_image", None):
                msg.base64_image = None
            if msg.role == "tool" and getattr(msg, "name", "") == "url_to_markdown":
                if last_assistant_idx > i:
                    msg.content = "[url_to_markdown output omitted from history]"
                    msg.base64_image = None
                continue
            if (not keep_dom) and msg.role == "user":
                try:
                    content = str(msg.content or "")
                    if "**Interactive Elements:**" in content:
                        import re

                        pattern = r"(\\*\\*Interactive Elements:\\*\\*)[\\s\\S]*?(?=\\n-{10,}|\\Z)"
                        msg.content = re.sub(pattern, r"\\1\\n[DOM removed from history]", content)
                except Exception:
                    pass

    async def format_next_step_user_prompt(self, session_id: str):
        state_rsp = await self._tool.get_current_state(session_id)
        if state_rsp.error:
            return
        browser_state: dict = json.loads(state_rsp.output)
        if not browser_state or browser_state.get("error"):
            return

        url = browser_state.get("url", "N/A")
        title = browser_state.get("title", "N/A")
        tabs = browser_state.get("tabs", [])
        if tabs:
            max_tabs = 8
            tab_lines = []
            for t in tabs[:max_tabs]:
                page_id = t.get("page_id", "N/A")
                tab_url = t.get("url", "N/A")
                tab_title = t.get("title", "N/A")
                current_mark = " *current*" if tab_url == url else ""
                tab_lines.append(f"- [{page_id}]{current_mark} {tab_title} — {tab_url}")
            more = f"\\n... (+{len(tabs) - max_tabs} more)" if len(tabs) > max_tabs else ""
            tabs_info = f"{len(tabs)} tabs open\\n" + "\\n".join(tab_lines) + more
        else:
            tabs_info = "No other tabs"

        interactive_elements = browser_state.get("interactive_elements", "No interactive elements found")
        results_info = self._last_action_results or ""
        max_len = 2000
        if len(results_info) > max_len:
            results_info = results_info[:max_len] + "\\n...[truncated]..."
        content = BROWSER_NEXT_STEP_PROMPT.format(
            url_placeholder=url,
            title_placeholder=title,
            tabs_placeholder=tabs_info,
            elements_placeholder=interactive_elements,
            results_placeholder=results_info,
        )
        b64_img = None
        if state_rsp.base64_image and self._tool._property.use_image:
            b64_img = self._process_image(state_rsp.base64_image)
        user_message = Message.user_message(content=content, base64_image=b64_img)
        self.memory.add_message(user_message)

    async def format_next_may_terminate_prompt(self, session_id: str):
        self.memory.add_message(
            Message.user_message(
                content=(
                    "If you want to stop interaction, use `terminate` tool/function call.\n\n"
                    "IMPORTANT: Screenshots and `url_to_markdown` outputs are only available for the current turn and will NOT be kept in later turns. "
                    "If the previous step used `url_to_markdown` or relied on a screenshot, summarize the key facts you extracted in your next assistant message."
                )
            )
        )

    async def initialize_session(self):
        await self._tool.create_session(self._session_id)
        self._finished = False
        self._current_system_prompt = TASK_EXECUTOR_SYSTEM_PROMPT
        self._message_queue = asyncio.Queue()
        self._last_content = ""
        self._resolve_trace_dirs()
        if self._trace_file_path and self._trace_images_dir:
            self._tracer = Tracer(file_path=self._trace_file_path, images_dir=self._trace_images_dir).init(mode="a")
            self._tracer.trace(
                {
                    "event": "request_start",
                    "request_id": self._session_id,
                    "question": self._question,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

    def is_finished(self) -> bool:
        return self._finished

    def final_result(self) -> str:
        if self._finish_reason == "unknown":
            self._finish_reason = "success"
        return self._last_content

    async def cleanup_session(self):
        await self._tool.destroy_session(self._session_id)
        if self._tracer:
            try:
                self._tracer.trace(
                    {
                        "event": "request_end",
                        "request_id": self._session_id,
                        "finish_reason": self._finish_reason,
                        "timestamp": datetime.datetime.now().isoformat(),
                    }
                )
            except Exception:
                pass
            self._tracer.exit()

    async def put_message(self, msg):
        await self._message_queue.put(msg)

    async def execute(self):
        self._current_step += 1
        self._clean_memory()
        think_result = await self._reasoning()
        LOGGER.info("should act: %s, content: %s", think_result.should_act, think_result.content)
        if not self._last_content:
            self._last_content = think_result.content
        if not think_result.should_act:
            self._last_content = think_result.content
            return "Thinking complete - no action needed"
        result = await self._acting(think_result)
        await self.put_message("STEP_DONE")
        return result

    async def final_execute(self):
        tool_rsp = await self._tool.get_current_content(self._session_id)
        return tool_rsp.output if tool_rsp.output else tool_rsp.error

    async def _reasoning(self) -> ReasoningResult:
        ret = ReasoningResult(should_act=False)
        try:
            def _strip_unsupported_schema_fields(obj: Any, keys_to_remove: set) -> Any:
                if isinstance(obj, dict):
                    new_obj = {}
                    for k, v in obj.items():
                        if k in keys_to_remove:
                            continue
                        new_obj[k] = _strip_unsupported_schema_fields(v, keys_to_remove)
                    return new_obj
                if isinstance(obj, list):
                    return [_strip_unsupported_schema_fields(x, keys_to_remove) for x in obj]
                return obj

            tools_payload = self._available_tools.to_param()
            model_name = str(getattr(self._llm, "model", "") or "").lower()
            if "gemini" in model_name:
                tools_payload = _strip_unsupported_schema_fields(copy.deepcopy(tools_payload), keys_to_remove={"dependencies"})

            response = await self._llm.ask_tool(
                self.memory.get_recent_messages(10000),
                system_msgs=([Message.system_message(self._current_system_prompt)] if self._current_system_prompt else None),
                tools=tools_payload,
                tool_choice=self._tool_choice,
                tracer=self._tracer,
            )
        except Exception as e:  # pylint: disable=broad-except
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                await self.put_message(f"🚨 Token limit error (from RetryError): {token_limit_error}")
                self.memory.add_message(Message.assistant_message(f"Maximum token limit reached, cannot continue execution: {str(token_limit_error)}"))
                self._finish_reason = "token_limit"
                self._finished = True
                ret.should_act = False
                return ret
            raise

        raw_content = response.content if response and response.content else ""
        ret.content = raw_content
        ret.tool_calls = response.tool_calls if response and response.tool_calls else []
        await self.put_message(f"✨ {self.name}'s thoughts: {ret.content}")
        await self.put_message(f"🛠️ {self.name} selected {len(ret.tool_calls) if ret.tool_calls else 0} tools to use")

        assistant_msg = (
            Message.from_tool_calls(content=ret.content, tool_calls=ret.tool_calls)
            if ret.tool_calls
            else Message.assistant_message(ret.content)
        )
        self.memory.add_message(assistant_msg)

        if self._tool_choice == ToolChoice.REQUIRED and not ret.tool_calls:
            ret.should_act = True
            return ret
        if self._tool_choice == ToolChoice.AUTO and not ret.tool_calls:
            ret.should_act = bool(ret.content)
            return ret
        ret.should_act = bool(ret.tool_calls)
        return ret

    async def _acting(self, thinking: ReasoningResult):
        if not thinking.tool_calls:
            if self._tool_choice == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)
            if self._finish_reason == "unknown":
                self._finish_reason = "no_tools_called"
            return self.memory.messages[-1].content or "No content or commands to execute"

        results = []
        tool_calls = thinking.tool_calls or []
        if len(tool_calls) > 1:
            await self.put_message(f"⚠️ Tool-call limit enforced: received {len(tool_calls)} tool calls, executing only the first.")

        for command in tool_calls[:1]:
            result = await self._execute_tool(command)
            tool_output = result.output if result.output else result.error
            await self.put_message(f"🎯 Tool '{command.function.name}' completed its mission! Result: {tool_output}")

            tool_msg = Message.tool_message(
                content=tool_output,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=result.base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(tool_output)

        self._last_action_results = "\n\n".join(results)
        tool_name_set = set([call.function.name for call in thinking.tool_calls])
        if len(tool_name_set) == 0:
            self._finished = True
            self._finish_reason = "no_tools_called"
            if thinking.content.strip():
                self._last_content = thinking.content.strip()
            return "Thinking complete - no action needed"

        if "browser_use" in tool_name_set:
            await self.format_next_step_user_prompt(self._session_id)
        else:
            await self.format_next_may_terminate_prompt(self._session_id)
            if thinking.content.strip():
                self._last_content = thinking.content.strip()

        if "terminate" in tool_name_set:
            self._finished = True
            self._finish_reason = "terminated"
        else:
            self._last_content = thinking.content.strip()
        return "\n\n".join(results)

    async def _execute_tool(self, command: ToolCall) -> ToolResult:
        if not command or not command.function or not command.function.name:
            return ToolResult(error="Error: Invalid command format")
        name = command.function.name
        if not self._available_tools.has(name):
            return ToolResult(error=f"Error: Unknown tool '{name}'")
        try:
            args = json.loads(command.function.arguments or "{}")
            await self.put_message(f"🔧 Activating tool: '{name}'...")
            result: ToolResult = await self._available_tools.execute(name=name, args=args, session_id=self._session_id)
            if result.error:
                observation = f"⚠️ Tool '{name}' encountered a problem: {result.error}"
            else:
                observation = f"Observed output of cmd `{name}` executed:\n{result.output}" if result.output else f"Cmd `{name}` completed with no output"
            return ToolResult(output=observation, base64_image=result.base64_image)
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            LOGGER.exception(error_msg)
            return ToolResult(error=f"Error: {error_msg}")
        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            LOGGER.exception(error_msg)
            return ToolResult(error=f"Error: {error_msg}")

    def _update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }
        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))


