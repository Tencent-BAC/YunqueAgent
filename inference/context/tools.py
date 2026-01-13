"""
A powerful context management module that helps to generate and manage working memory based on tools interactions and responses. 
"""

import os
import asyncio
import json
import logging
import traceback
from openai import OpenAI
from typing import List, Optional
from context.prompt import CONTEXT_SYSTEM_PROMPT, CONTEXT_PROMPT
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class LLM:
    def __init__(self, model: str, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        # Use OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=600.0,
        )
    def ask(self, messages: List[dict], temperature: float, max_tokens: int, response_format: dict):
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        message = chat_response.choices[0].message
        reasoning_content = getattr(message, 'reasoning_content', "") or ""
        raw_content = message.content
        if isinstance(raw_content, list):
            content = raw_content[0].get('text', '') if raw_content else ""
        else:
            content = raw_content or ""
        return content


class MemoryUnit(BaseModel):
    """Structural memory unit"""
    rounds_index: List[int] = Field(description="indexes of folded rounds")
    sub_goal: str = Field(description="sub goal")  # subgoal
    tools_log: List[dict] = Field(description="tools log")
    summary: str = Field(description="memory summary")


class MemoryList:
    """All memory units in each rollout"""
    def __init__(self, max_memory_num = 100):
        self._memory_units: List[MemoryUnit] = []
        self.max_memory_num: int = max_memory_num  # max number of memory units

    def clear_memory(self):
        """Clear all memories for this session."""
        self._memory_units.clear()

    def add_unit(self, round_index: int, unit: dict) -> None:
        """Add a new memory unit"""
        self._memory_units.append(
            MemoryUnit(
                rounds_index=[round_index], 
                sub_goal=unit["sub_goal"],
                tools_log=unit["tools_log"],
                summary=unit["summary"],
            )
        )
        if len(self._memory_units) > self.max_memory_num:
            self._memory_units = self._memory_units[-self.max_memory_num:]

    def update_latest_unit(self, round_index: int, unit: dict):
        """Update the latest memory unit"""
        latest_unit = self.get_latest_unit()
        # append latest round index
        latest_unit.rounds_index.append(round_index)
        # update tools_log and summary
        latest_unit.tools_log = unit["tools_log"]
        latest_unit.summary = unit["summary"]
        # replace the latest memory unit
        self._memory_units[-1] = latest_unit

    def get_latest_unit(self) -> MemoryUnit:
        """Get the latest memory"""
        if len(self._memory_units) == 0:
            return
        return self._memory_units[-1]

    def get_all_units(self) -> List[MemoryUnit]:
        """Get all of memories"""
        return self._memory_units

    def get_units_num(self) -> int:
        """Get number of memory units"""
        return len(self._memory_units)


class ContextManageTool(object):

    def __init__(self, llm: LLM, max_memory_num: int = 100):
        self.llm = llm
        self.system_prompt: str = CONTEXT_SYSTEM_PROMPT
        self.prompt: str = CONTEXT_PROMPT
        self.memory_list = MemoryList(max_memory_num)

    def _gen_memory(
        self, query, latest_observation, latest_response, latest_memory, retry_num=3
    ):
        """Generate new memory unit"""
        user_prompt = (
            f"{self.prompt}\n"
            f"User Question: {query}\n"
            f"Recent Memory: {latest_memory}\n"
            f"Latest Agent Response: {latest_response}\n"
            f"Tool Response: {latest_observation}\n"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for i in range(retry_num):
            try:
                response = self.llm.ask(
                    messages=messages,
                    temperature=float(os.getenv("MEMORY_TEMPERATURE", 0.9)),
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                )
                merge = int(json.loads(response)["merge"])
                memory = json.loads(response)["memory"]
                return merge, memory
            except Exception as e:  # pylint: disable=broad-except
                if i + 1 >= retry_num:
                    raise e
                else:
                    LOGGER.debug(f"warning:{e}, retrying...{i + 1}/{retry_num}")


    def update_memory(self, query, round_index, latest_observation, latest_response):
        """
        Execute a memory update action. This method includes:
        1. Generate new memory unit based on the latest observation and response.
        2. Update the memory list.
        3. Return the latest unit of memory list.
        """
        latest_memory_unit = self.memory_list.get_latest_unit()
        if not latest_memory_unit:
            latest_memory_unit = ""
        else:
            latest_memory_unit = json.dumps(latest_memory_unit.model_dump(), ensure_ascii=False)
        
        try:
            is_merged, new_memory_unit = self._gen_memory(
                query, latest_observation, latest_response, latest_memory_unit
            )
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error(f"error:{e}, traceback:{traceback.format_exc()}")
            return

        if is_merged == 0 or not latest_memory_unit:
            self.memory_list.add_unit(
                round_index, new_memory_unit
            )
        elif is_merged == 1:
            self.memory_list.update_latest_unit(
                round_index, new_memory_unit
            )
        else:
            LOGGER.error(
                f"[update memory] Update Memory List Error: is_merged = {is_merged}"
            )
            return
        latest_memory_unit = self.memory_list.get_latest_unit().model_dump()
        return json.dumps(latest_memory_unit, ensure_ascii=False)

    def get_memory(self):
        """Get memory list"""
        try:
            memory_list = [
                m.model_dump() for m in self.memory_list.get_all_units()
            ]
            return json.dumps(memory_list, ensure_ascii=False)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.error(f"error:{e}, traceback:{traceback.format_exc()}")


def create_context_manage_tool():
    def create_llm():
        """Create LLM instance based on configuration."""
        return LLM(
            os.getenv("MEMORY_MODEL", None), 
            os.getenv("MEMORY_API_KEY", None), 
            os.getenv("MEMORY_API_BASE", None),
        )

    return ContextManageTool(llm=create_llm())
