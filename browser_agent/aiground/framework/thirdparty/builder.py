# -*- coding: utf-8 -*-
"""
builder

Third-party component builder utilities.
"""


import copy
from typing import Dict, List

from aiground.common.dict_args import DictArgs

from .browser_use.browser.browser import Browser, BrowserConfig, ProxySettings
from .openmanus.app.agent.manus import Manus
from .openmanus.app.config import BrowserSettings, LLMSettings, MCPServerConfig
from .openmanus.app.llm import LLM
from .openmanus.app.tool.ask_human import AskHuman
from .openmanus.app.tool.browser_use_tool import BrowserUseTool
from .openmanus.app.tool.python_execute import PythonExecute
from .openmanus.app.tool.str_replace_editor import StrReplaceEditor
from .openmanus.app.tool.terminate import Terminate
from .openmanus.app.tool.tool_collection import ToolCollection


class Builder(object):
    """Builder class."""

    @classmethod
    async def create_manus(cls, config: DictArgs) -> Manus:
        """Create a Manus instance."""
        kwargs = config.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        llm_config = config.llm_config
        llm = cls.create_llm(llm_config)
        kwargs["llm"] = llm

        tools_config = config.tools
        tools = cls.create_tools(tools_config)
        kwargs["available_tools"] = tools

        manus = Manus(**kwargs)
        servers_config: dict = config.mcp_config.servers
        server_settings: Dict[str, MCPServerConfig] = {}
        for server_name, server_config in servers_config.items():
            server_settings[server_name] = MCPServerConfig.model_validate(
                server_config, strict=False
            )
        await manus.init_mcp_servers_with(servers=server_settings)
        return manus

    @classmethod
    def create_llm(cls, llm_config: DictArgs):
        """Create an LLM instance."""
        final_llm_config = {}
        for k, v in llm_config.items():
            if not v or not isinstance(v, dict):
                continue
            final_llm_config[k] = LLMSettings.model_validate(v, strict=False)
        return LLM(llm_config.name, final_llm_config)

    @classmethod
    def create_browser_use(cls, browser_config: BrowserSettings):
        browser_config_kwargs = {"headless": False, "disable_security": True}

        # handle proxy settings.
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

        browser = Browser(BrowserConfig(**browser_config_kwargs))
        return browser

    @classmethod
    def create_tools(cls, tools_config: List[DictArgs]):
        """Create tool list."""
        tools = []
        for tool_config in tools_config:
            tool = cls.create_tool(tool_config)
            if tool is not None:
                tools.append(tool)
        return ToolCollection(*tools)

    @classmethod
    def create_tool(cls, tool_config: DictArgs):
        """Create a tool:
        BrowserUseTool()
        StrReplaceEditor()
        AskHuman()
        Terminate()
        """
        tool_name = tool_config.name
        if tool_name == "python_execute":
            return PythonExecute()
        elif tool_name == "browser_use":
            kwargs = tool_config.kwargs
            if kwargs is None:
                kwargs = {}
            kwargs = copy.deepcopy(kwargs)
            if hasattr(tool_config, "browser_config"):
                browser_config = tool_config.get("browser_config")
                browser_settings = BrowserSettings.model_validate(
                    browser_config, strict=False
                )
                # browser, context, dom_service = await cls.create_browser_use(
                #     browser_settings
                # )
                browser = cls.create_browser_use(browser_settings)
                kwargs["browser"] = browser
                # kwargs["context"] = context
                # kwargs["dom_service"] = dom_service
            if hasattr(tool_config, "llm_config"):
                llm_config = tool_config.get("llm_config")
                llm = cls.create_llm(llm_config)
                kwargs["llm"] = llm
            return BrowserUseTool(**kwargs)
        elif tool_name == "str_replace_editor":
            return StrReplaceEditor()
        elif tool_name == "ask_human":
            return AskHuman()
        elif tool_name == "terminate":
            return Terminate()
        return None


if __name__ == "__main__":
    pass
