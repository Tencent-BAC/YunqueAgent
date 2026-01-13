"""
Agent Supervisor - 负责监督 Agent 的输出质量并在需要时进行反思纠正
"""

import os
from typing import Dict, List, Optional, Tuple


class Supervisor:
    """
    Agent 监督器，负责检测和修正 Agent 的输出问题。
    当 ENABLE_REFLECTION 开启时，在以下场景提供反思和纠正：
    1. 响应被截断（超过长度限制）
    2. 缺少工具调用或最终答案
    3. 接近 LLM 调用次数限制
    4. 超过 token 上下文限制
    """
    
    def __init__(self, enable_reflection: Optional[bool] = None, llm_caller=None):
        """
        初始化监督器
        
        Args:
            enable_reflection: 是否启用反思功能，默认从环境变量读取
            llm_caller: LLM 调用函数，用于在 supervisor 内部调用模型
        """
        if enable_reflection is None:
            enable_reflection = os.getenv('ENABLE_REFLECTION', '1').strip().lower() in ('1', 'true', 'yes', 'y', 'on')
        self.enable_reflection = enable_reflection
        self.llm_caller = llm_caller
        print(f"\n[Supervisor] Reflection mode: {'ENABLED' if self.enable_reflection else 'DISABLED'}\n")
    
    def handle_truncated_response(
        self, 
        reasoning_content: str, 
        messages: List[Dict]
    ) -> Tuple[bool, List[Dict], str]:
        """
        处理被截断的响应（思维链过长导致超过 max_tokens）
        
        Args:
            reasoning_content: 推理内容
            messages: 当前消息列表
            
        Returns:
            (should_retry, updated_messages, action_description)
            - should_retry: 是否应该重试
            - updated_messages: 更新后的消息列表
            - action_description: 执行的动作描述
        """
        if not self.enable_reflection:
            return False, messages, "Reflection disabled. Accepting truncated response."
        
        if not reasoning_content or not reasoning_content.strip():
            return False, messages, "No reasoning content to reflect on."
        
        correction_message = {
            "role": "user",
            "content": "Your previous response was truncated since the reasoning process was highly repetitive or excessive. Please analyze the most critical information in the original history, re-evaluate your strategy, and provide a more concise response."
        }
        
        updated_messages = messages.copy()
        updated_messages.append(correction_message)
        
        return True, updated_messages, "Adding correction prompt to request more concise response."
    
    def handle_missing_action(self, messages: List[Dict]) -> Tuple[bool, List[Dict], str]:
        """
        处理缺少工具调用或最终答案的情况
        
        Args:
            messages: 当前消息列表
            
        Returns:
            (should_retry, updated_messages, action_description)
        """
        if not self.enable_reflection:
            return False, messages, "Reflection disabled. Continuing without re-prompting."
        
        # 修改最后一条消息（应该是 assistant 的响应）
        updated_messages = messages.copy()
        if updated_messages:
            updated_messages[-1]['content'] = (
                "You did not make any tool calls or give the final answer. "
                "If you need more information, please call an available tool in the format: "
                "<think>your thinking</think>\n<tool_call>your tool call</tool_call>. "
                "If you think you have enough information to answer the question, "
                "please provide your final answer in the format: "
                "<think>your final thinking</think>\n<answer>your answer</answer>"
            )
        
        return True, updated_messages, "Prompting model to make tool call or provide final answer."
    
    def handle_approaching_limit(
        self, 
        messages: List[Dict], 
        calls_remaining: int
    ) -> Tuple[bool, Optional[Dict], str]:
        """
        处理接近 LLM 调用次数限制的情况
        
        Args:
            messages: 当前消息列表
            calls_remaining: 剩余可用调用次数
            
        Returns:
            (should_intervene, result_dict, action_description)
            - should_intervene: 是否需要介入
            - result_dict: 如果不启用反思，返回结果字典；否则返回 None
            - action_description: 执行的动作描述
        """
        if not self.enable_reflection:
            return True, {
                "prediction": "LLM call limit reached",
                "termination": "llm call limit reached"
            }, "Reflection disabled. Stopping without final prompt."
        
        # 修改最后一条消息，引导模型给出最终答案
        if messages:
            messages[-1]['content'] = (
                " You are approaching the maximum number of LLM calls allowed. "
                "You should stop making tool calls and, based on all the information above, "
                "think again and provide what you consider the most likely answer in the following format:"
                "<think>your final thinking</think>\n<answer>your answer</answer>"
            )
        
        return False, None, f"Prompting model for final answer (calls remaining: {calls_remaining})."
    
    def handle_approaching_timeout(
        self,
        messages: List[Dict],
        elapsed_time: float,
        timeout_minutes: float
    ) -> Tuple[bool, Optional[Dict], str]:
        """
        处理接近超时的情况
        
        Args:
            messages: 当前消息列表
            elapsed_time: 已经过的时间（秒）
            timeout_minutes: 超时限制（分钟）
            
        Returns:
            (should_intervene, result_dict, action_description)
            - should_intervene: 是否需要介入
            - result_dict: 如果不启用反思，返回结果字典；否则返回 None
            - action_description: 执行的动作描述
        """
        if not self.enable_reflection:
            return True, {
                "prediction": "Approaching timeout",
                "termination": "approaching timeout"
            }, "Reflection disabled. Stopping without final prompt."
        
        # 修改最后一条消息，引导模型给出最终答案
        if messages:
            elapsed_minutes = elapsed_time / 60
            remaining_minutes = timeout_minutes - elapsed_minutes
            messages[-1]['content'] = (
                f" You are approaching the timeout limit (elapsed: {elapsed_minutes:.1f} min, "
                f"timeout: {timeout_minutes:.0f} min, remaining: {remaining_minutes:.1f} min). "
                "You should stop making tool calls and, based on all the information above, "
                "think again and provide what you consider the most likely answer in the following format:"
                "<think>your final thinking</think>\n<answer>your answer</answer>"
            )
        
        return False, None, f"Prompting model for final answer (time: {elapsed_time/60:.1f}/{timeout_minutes:.0f} min)."
    
    def handle_token_limit_exceeded(
        self, 
        messages: List[Dict], 
        token_count: int, 
        max_tokens: int
    ) -> Tuple[bool, Optional[Dict], str]:
        """
        处理超过 token 上下文限制的情况
        
        Args:
            messages: 当前消息列表
            token_count: 当前 token 数量
            max_tokens: 最大 token 限制
            
        Returns:
            (should_intervene, result_dict, action_description)
        """
        if not self.enable_reflection:
            return True, {
                "prediction": "Token limit reached",
                "termination": "token limit reached"
            }, "Reflection disabled. Stopping without final prompt."
        
        # 修改最后一条消息，强制模型给出答案
        if messages:
            messages[-1]['content'] = (
                "You have now reached the maximum context length you can handle. "
                "You should stop making tool calls and, based on all the information above, "
                "think again and provide what you consider the most likely answer in the following format:"
                "<think>your final thinking</think>\n<answer>your answer</answer>"
            )
        
        return False, None, f"Prompting model for final answer (tokens: {token_count}/{max_tokens})."
    
    def parse_final_response(
        self, 
        content: str
    ) -> Tuple[str, str]:
        """
        解析最终响应，提取答案和终止原因
        
        Args:
            content: 模型响应内容
            
        Returns:
            (prediction, termination_reason)
        """
        if '<answer>' in content and '</answer>' in content:
            try:
                prediction = content.split('<answer>')[1].split('</answer>')[0]
                return prediction, "answer"
            except IndexError:
                return content, "format error: answer tag parsing failed"
        
        return content, "format error: no answer tag found"
    
    def handle_tool_parsing_error(
        self,
        error_message: str,
        tool_call_content: str,
        messages: List[Dict]
    ) -> Optional[str]:
        """
        处理工具调用解析错误，在内部调用 LLM 获取纠正后的响应
        
        Args:
            error_message: 错误信息
            tool_call_content: 原始工具调用内容
            messages: 当前消息列表
            
        Returns:
            new_content: 纠正后的新响应内容，如果失败返回 None
        """
        if not self.enable_reflection:
            print(f"[Supervisor] Reflection disabled. Tool parsing error: {error_message}")
            return None
        
        if not self.llm_caller:
            print("[Supervisor] LLM caller not available.")
            return None
        
        # 构建纠错提示，参考 prompt.py 中的格式说明
        correction_prompt = (
            f"Your previous tool call could not be parsed due to the following error:\n"
            f"{error_message}\n\n"
            f"Original tool call content:\n{tool_call_content}\n\n"
            f"Please correct the format and regenerate the tool call. Remember:\n\n"
            f"**For CodeExecutor tool:**\n"
            f"The 'arguments' JSON object must be empty: {{}}.\n"
            f"The Python code must be placed immediately after the JSON block, enclosed within <code> and </code> tags.\n\n"
            f"Example of a correct CodeExecutor call:\n"
            f"<tool_call>\n"
            f'{{"name": "CodeExecutor", "arguments": {{}}}}\n'
            f"<code>\n"
            f"import numpy as np\n"
            f"# Your code here\n"
            f"print(f\"The result is: {{np.mean([1,2,3])}}\")\n"
            f"</code>\n"
            f"</tool_call>\n\n"
            f"**For other tools (search, visit, google_scholar):**\n"
            f"Use standard JSON format within <tool_call></tool_call> tags:\n"
            f"<tool_call>\n"
            f'{{"name": "<function-name>", "arguments": <args-json-object>}}\n'
            f"</tool_call>\n\n"
            f"Please regenerate your tool call with the correct format now."
        )
        
        # 准备临时消息
        temp_messages = messages.copy()

        # 添加纠错提示
        temp_messages.append({
            "role": "user",
            "content": correction_prompt
        })
        
        # 在 supervisor 内部调用 LLM
        try:
            new_content = self.llm_caller(temp_messages, max_tries=3)
            print(f"[Supervisor] Successfully obtained corrected response from LLM")
            return new_content
        except Exception as e:
            print(f"[Supervisor] Failed to get corrected response: {e}")
            return None
    
    def format_limit_reached_result(
        self,
        content: str,
        limit_type: str  # "llm_call" or "token"
    ) -> Tuple[str, str]:
        """
        格式化达到限制时的结果
        
        Args:
            content: 模型响应内容
            limit_type: 限制类型 ("llm_call" 或 "token")
            
        Returns:
            (prediction, termination_reason)
        """
        if '<answer>' in content and '</answer>' in content:
            try:
                prediction = content.split('<answer>')[1].split('</answer>')[0]
                return prediction, f"generate an answer as {limit_type} limit reached"
            except IndexError:
                return content, f"format error: generate an answer as {limit_type} limit reached"
        else:
            return content, f"format error: generate an answer as {limit_type} limit reached"
