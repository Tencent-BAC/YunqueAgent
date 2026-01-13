"""
通过 BrowserUse MCP 服务执行真实浏览器访问，并返回摘要与推理轨迹。
负责构建任务描述、调用流式接口并解析事件。
"""

import json
import os
import re
import threading
import time
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor
import requests
import tiktoken
from file_tools.file_parser import SingleFileParser
from file_tools.utils import get_basename_from_url
from tool_file import FileParser
from openai import OpenAI
from prompt import EXTRACTOR_PROMPT
from base_tool import BaseTool, register_tool

# BrowserUse MCP 服务地址，指向dev cloud机器
DEFAULT_BROWSER_AGENT_ENDPOINT = "http://127.0.0.1:9091/browser_use_agent"
BROWSER_AGENT_ENDPOINT = os.getenv("BROWSER_AGENT_ENDPOINT", DEFAULT_BROWSER_AGENT_ENDPOINT)
# 是否启用 browser agent（不启用时，仅使用 Jina 抓取+摘要 web URL）
ENABLE_BROWSER_AGENT = os.getenv("ENABLE_BROWSER_AGENT", "1").strip().lower() in ("1", "true", "yes", "y", "on")
# 复用 VISIT_SERVER_TIMEOUT，以便统一控制整体访问时间。
BROWSER_AGENT_TIMEOUT = int(os.getenv("BROWSER_AGENT_TIMEOUT", os.getenv("VISIT_SERVER_TIMEOUT", "200")))
# 限制浏览器agent步数，避免长链路挂起。
BROWSER_AGENT_MAX_STEPS = int(os.getenv("BROWSER_AGENT_MAX_STEPS", "10"))
STREAM_PREFIX = "data:"
# 是否启用数据分析能力（针对文件类 URL）
ENABLE_DEEP_ANALYSIS = os.getenv("ENABLE_DEEP_ANALYSIS", "0").strip().lower() in ("1", "true", "yes", "y", "on")
# 是否启用反思和 fallback 机制
ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "1").strip().lower() in ("1", "true", "yes", "y", "on")

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
EXTRACTOR_PROMPT_SHORT = EXTRACTOR_PROMPT.replace(
    ", it can be more than three paragraphs.",
    ". If the source material is extensive, limit the extraction to the most critical paragraphs or sentences.",
)
EXTRACTOR_PROMPT_SUMMARY_ONLY = EXTRACTOR_PROMPT.replace(
    "Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.",
    "Skip this step and output 'No evidence content available, please refer to the summary.' instead.",
)


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def clean_invalid_escape(text: str) -> str:
    """
    Valid JSON escape characters ('"', '\', '/', 'b', 'f', 'n', 'r', 't', 'u' hex hex hex hex)
    """
    text = text.replace("\\t", "").replace("\\v", "")
    pattern = r"\\(?![\\\"/bfnru])|\\$"
    return re.sub(pattern, "", text)


@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    """将 visit 工具代理到 BrowserUse MCP 服务器的封装类。"""

    name = "visit"
    # description = 'Visit webpage(s) and return the summary of the content.'
    description = "Visit webpage(s) via the BrowserUse MCP server and return the summary."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal or question for visiting the page(s)."
            },
            "max_steps": {
                "type": "integer",
                "description": "Optional override for browser agent step limit."
            }
        },
        "required": ["url", "goal"]
    }

    def __init__(self):
        super().__init__()
        self._trace_lock = threading.Lock()
        self._trace_by_thread = {}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """解析调用参数并触发浏览器代理任务。"""
        try:
            url = params["url"]
            goal = params["goal"]
            # question
            question = params.get("question", "")
            sub_goal = params["params"]["goal"] if "params" in params and "goal" in params["params"] else ""
        except Exception:
            return "[visit] Invalid request format: please provide 'url' and 'goal'."

        max_steps = params.get("max_steps", BROWSER_AGENT_MAX_STEPS)
        enable_browser_agent = ENABLE_BROWSER_AGENT
        url_list = self._ensure_url_list(url)
        start_time = time.time()
        
        # Initialize lists for file URLs and web URLs
        file_urls = []
        web_urls = []
        responses = []
        
        for target_url in url_list:
            try:
                url_type_info = FileParser.get_url_type_by_get_request(target_url)
                print(f"[visit] URL Type Info: {url_type_info} ======> URL: {target_url}", flush=True)
                if url_type_info.get('url_type') == 'file' and (url_type_info.get('file_type') == 'zip' or url_type_info.get('file_type') == 'csv' or url_type_info.get('file_type') == 'txt'):
                    file_urls.append(target_url)
                else:
                    web_urls.append(target_url)
            except Exception as e:
                print(f"[visit] Error checking URL type for {target_url}: {e}. Treating as web page.")
                web_urls.append(target_url)
                
        # File URLs are processed with deep analysis agent (if enabled)
        if file_urls:
            if not ENABLE_DEEP_ANALYSIS:
                # Fallback: treat file URLs as web URLs when deep analysis is disabled
                print(f"[visit] Deep analysis disabled. Treating file URLs as web URLs: {file_urls}")
                web_urls.extend(file_urls)
            else:
                from deep_analysis_agent import DeepAnalyzer
                deep_analyzer = DeepAnalyzer()
                print(f"[visit] Processing file URLs with DeepAnalyzer: {file_urls}")
                try:
                    for file_url in file_urls:
                        if time.time() - start_time > 900:
                            print(f"[visit] Timeout: Total visit time exceeded 15 minutes. Visiting {file_url}.")
                            cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=file_url, goal=goal)
                            cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                            cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                            continue
                            
                        # Get URL type info first before downloading
                        url_type_info = FileParser.get_url_type_by_get_request(file_url)
                        download_info = deep_analyzer.download(url_type_info)
                        print(f"[visit] Download info for {file_url}: {download_info}")
                        if download_info.get('status') != 'success':
                            cur_response = f"Error downloading {file_url}: {download_info.get('error_message', 'Unknown error')}"
                            print(f"[visit] Error downloading {file_url}: {download_info.get('error_message', 'Unknown error')}")
                        else:
                            cur_response = deep_analyzer.call(download_info, sub_goal, question)
                        responses.append(cur_response)
                except Exception as e:
                    print(f"[visit] Error processing file URLs with DeepAnalyzer: {e}")
                
        # Web URLs are processed with browser agent and Jina
        if web_urls:
            url_desc = self._normalize_url(url)
            task = f'Question: "{goal}". URL: {url_desc}'

            if enable_browser_agent:
                # 并发请求 browser agent 与 Jina 摘要，降低等待时间。
                with ThreadPoolExecutor(max_workers=2) as executor:
                    jina_future = executor.submit(
                        self._summarize_urls_with_jina, url_list, goal, start_time
                    )
                    agent_future = executor.submit(self._invoke_browser_agent, task, max_steps)

                    try:
                        agent_code, browser_payload, agent_trace = agent_future.result()
                    except Exception as exc:  # pylint: disable=broad-except
                        agent_code, browser_payload, agent_trace = None, f"[visit] Browser agent request failed: {exc}", {
                            "error": str(exc),
                            "events": [],
                            "task": task,
                        }

                    jina_result = jina_future.result()

                self._set_last_trace(agent_trace, goal, url_desc, max_steps)

                if agent_code == 200:
                    # 拼接 browser agent 与 Jina 的结果。
                    combined = []
                    if browser_payload:
                        combined.append(f"[Browser Agent]\n{browser_payload.strip()}")
                    if jina_result:
                        combined.append(f"[Jina]\n{jina_result.strip()}")
                    web_response = "\n\n".join([part for part in combined if part]) or "[visit] Browser agent returned empty response."
                    responses.append(web_response)
                else:
                    extra_tab_urls = self._parse_extra_tab_urls(browser_payload)
                    target_urls = self._merge_urls(url_list, extra_tab_urls)
                    if target_urls:
                        # 201 或其他非 200，继续使用 Jina 方案；已对原始 URL 做过一次，补充 extra tabs。
                        remaining_urls = [u for u in target_urls if u not in url_list]
                        extra_result = ""
                        if remaining_urls:
                            extra_result = self._summarize_urls_with_jina(
                                remaining_urls, goal, start_time
                            )

                        stitched = []
                        if jina_result:
                            stitched.append(f"[Jina]\n{jina_result.strip()}")
                        if extra_result:
                            stitched.append(f"[Jina][extra_tabs]\n{extra_result.strip()}")
                        web_response = "\n=======\n".join(stitched).strip()
                        responses.append(web_response)
                    else:
                        # 没有可用 URL，则返回已有的 Jina 结果或提示。
                        web_response = (f"[Jina]\n{jina_result.strip()}" if jina_result else "") or (
                            "[visit] Browser agent returned no usable URLs."
                        )
                        responses.append(web_response)
            else:
                # 不启用 browser agent：仅使用 Jina 抓取+摘要 web URL。
                jina_result = self._summarize_urls_with_jina(url_list, goal, start_time)
                web_response = (f"[Jina]\n{jina_result.strip()}" if jina_result else "") or "[visit] Jina returned empty response."
                responses.append(web_response)
        
        final_response = "\n=======\n".join(responses).strip()
        print(f'Summary Length {len(final_response)}; Summary Content {final_response}')
        return final_response
          

    @staticmethod
    def _normalize_url(url: Union[str, List[str]]) -> str:
        if isinstance(url, str):
            return url
        if isinstance(url, list):
            return ", ".join(url)
        raise ValueError("url must be a string or list of strings")

    @staticmethod
    def _ensure_url_list(url: Union[str, List[str]]) -> List[str]:
        if isinstance(url, str):
            return [url.strip()] if url.strip() else []
        if isinstance(url, list):
            urls: List[str] = []
            for item in url:
                if isinstance(item, str) and item.strip():
                    urls.append(item.strip())
            return urls
        raise ValueError("url must be a string or list of strings")

    @staticmethod
    def _merge_urls(original: List[str], extra: List[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for collection in (original, extra):
            for candidate in collection:
                normalized = candidate.strip()
                if normalized and normalized not in seen:
                    merged.append(normalized)
                    seen.add(normalized)
        return merged

    @staticmethod
    def _parse_extra_tab_urls(payload: str) -> List[str]:
        if not payload:
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict) and isinstance(parsed.get("extra_tabs"), list):
            tabs = parsed.get("extra_tabs") or []
        elif isinstance(parsed, list):
            tabs = parsed
        else:
            tabs = [parsed]

        urls: List[str] = []
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            link = tab.get("url", "")
            if isinstance(link, str):
                normalized = link.strip()
                if normalized:
                    urls.append(normalized)
        return urls

    def _invoke_browser_agent(self, task: str, max_steps: int):
        """构造请求并消费 BrowserUse 流式事件。"""
        start_time = time.time()
        payload = {"task": task, "max_steps": max_steps}
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        trajectory_events: List[dict] = []
        response = requests.post(
            BROWSER_AGENT_ENDPOINT,
            headers=headers,
            json=payload,
            stream=True,
            timeout=BROWSER_AGENT_TIMEOUT,
        )
        response.raise_for_status()

        reasoning_trace: List[str] = []
        final_message: str = ""
        final_code = None

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line
            if line.startswith(STREAM_PREFIX):
                # SSE 流以 `data:` 开头，先去掉前缀再解析 JSON。
                line = line[len(STREAM_PREFIX):].strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            code = event.get("code")
            data = event.get("data")
            text = self._stringify(data)
            trajectory_events.append(
                {
                    "code": code,
                    "data": data,
                    "text": text,
                }
            )

            if code == 100 and text:
                # code=100 表示模型推理节点，记录推理轨迹。
                reasoning_trace.append(text)
            elif code in (200, 201):
                # 200/201 为最终回答或退出，直接返回。
                final_code = code
                final_message = text or ""
                break
        # import pdb; pdb.set_trace()
        if final_code is None:
            raise RuntimeError("Browser agent stream ended without terminal event.")

        cleaned_message = final_message.strip()
        end_time = time.time()
        print(f"[visit] Browser agent time taken: {end_time - start_time} seconds. Task: {task}")
        return final_code, cleaned_message, {
            "events": trajectory_events,
            "reasoning_trace": reasoning_trace,
            "final_code": final_code,
            "final_message": cleaned_message,
            "task": task,
            "max_steps": max_steps,
        }

    def _set_last_trace(self, trace: dict, goal: str, url_desc: str, max_steps: int):
        """缓存最近一次 browser agent 轨迹，按线程隔离，便于上层持久化。"""
        if trace is None:
            trace = {}
        meta = {
            "goal": goal,
            "url": url_desc,
            "max_steps": max_steps,
        }
        trace_with_meta = {**trace, **meta}
        tid = threading.get_ident()
        with self._trace_lock:
            self._trace_by_thread[tid] = trace_with_meta

    def get_last_trace(self) -> dict:
        """获取当前线程最近一次 browser agent 轨迹。"""
        tid = threading.get_ident()
        with self._trace_lock:
            trace = self._trace_by_thread.get(tid)
        # 返回一个浅拷贝，避免外部修改内部缓存
        return dict(trace) if trace else None

    def _summarize_urls_with_jina(self, urls: List[str], goal: str, start_time: float) -> str:
        """遍历 URL 列表并使用 Jina+LLM 摘要（共享整体超时 900s）。"""
        _start_time = time.time()
        responses: List[str] = []
        for target in urls:
            if time.time() - start_time > 900:
                responses.append(
                    "The useful information in {url} for user goal {goal} as follows: \n\n"
                    "Evidence in page: \nThe visit timed out.\n\n"
                    "Summary: \nThe visit timed out.\n\n".format(url=target, goal=goal)
                )
                continue
            try:
                responses.append(self.readpage_jina(target, goal))
            except Exception as exc:  # pylint: disable=broad-except
                responses.append(
                    "The useful information in {url} for user goal {goal} as follows: \n\n"
                    "Evidence in page: \nError: {error}\n\n"
                    "Summary: \nFailed to retrieve page content.\n\n".format(url=target, goal=goal, error=str(exc))
                )
        _end_time = time.time()
        print("[visit] _summarize_urls_with_jina time taken:", _end_time - _start_time)
        return "\n=======\n".join(responses).strip()

    @staticmethod
    def _stringify(data):
        """将 event data 安全转为字符串，避免后续拼接报错。"""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, ensure_ascii=False)
        except TypeError:
            return str(data)

    def call_server(self, msgs, max_retries=2):
        api_key = os.environ.get("SUMMARY_API_KEY")
        url_llm = os.environ.get("SUMMARY_API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        client = OpenAI(api_key=api_key, base_url=url_llm)
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7,
                )
                content = chat_response.choices[0].message.content
                finish_reason = chat_response.choices[0].finish_reason
                if finish_reason == "length":
                    if attempt == (max_retries - 1):
                        return "[summary] Response length exceeded limit."
                    continue
                if content:
                    try:
                        json.loads(content)
                    except Exception:
                        left = content.find("{")
                        right = content.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            content = content[left : right + 1]
                    return content
            except Exception:  # pylint: disable=broad-except
                if attempt == (max_retries - 1):
                    return ""
                continue
        return ""

    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.

        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        timeout = 50
        headers = {"Authorization": f"Bearer {JINA_API_KEYS}"}
        for attempt in range(max_retries):
            try:
                response = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
                if response.status_code == 200:
                    return response.text
                raise ValueError(f"jina readpage error: {response.status_code}")
            except Exception:  # pylint: disable=broad-except
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
        return "[visit] Failed to read page."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 3
        for _ in range(max_attempts):
            content = self.jina_readpage(url)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
            
        return "[visit] Failed to read page."

    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        start_time = time.time()
        summary_page_func = self.call_server
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
        
        content_jina = self.html_readpage_jina(url)

        content = f"""
        <source tool="jina">
        {content_jina}
        </source>
        """
        # TODO: pdf不能直接抓取，否则全是乱码
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            raw = summary_page_func(messages, max_retries=max_retries)

            if ENABLE_REFLECTION and isinstance(raw, str) and raw.startswith("[summary] Response length exceeded limit."):
                # If response length exceeded limit, retry with shorter evidence and summary context
                messages_short = [{"role": "user", "content": EXTRACTOR_PROMPT_SHORT.format(webpage_content=content, goal=goal)}]
                raw = summary_page_func(messages_short, max_retries=max_retries)
                # If still exceeds length, keep summary content only
                if isinstance(raw, str) and raw.startswith("[summary] Response length exceeded limit."):
                    messages_summary_only = [
                        {"role": "user", "content": EXTRACTOR_PROMPT_SUMMARY_ONLY.format(webpage_content=content, goal=goal)}
                    ]
                    raw = summary_page_func(messages_summary_only, max_retries=max_retries)

            if ENABLE_REFLECTION:
                summary_retries = 3
                while isinstance(raw, str) and len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    content = content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = summary_page_func(messages, max_retries=max_retries)
                    summary_retries -= 1

            raw_json = None
            parse_retry_times = 0
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 2:
                try:
                    raw_json = json.loads(raw)
                    break
                except Exception:
                    if not ENABLE_REFLECTION:
                        # 如果禁用 reflection，直接退出
                        break
                    
                    try:
                        # Retry after cleaning invalid escape characters
                        cleaned = clean_invalid_escape(raw)
                        raw_json = json.loads(cleaned)
                        break
                    except Exception:
                        pass
                    
                    # Retry using file parser for specific file types
                    f_type = get_basename_from_url(url).split(".")[-1].lower()
                    if f_type in ["pdf", "docx", "pptx", "txt", "html", "csv", "xlsx", "xls", "doc", "zip"]:
                        try:
                            result = SingleFileParser().call(json.dumps({"url": url}))
                            messages = [{"role": "user", "content": EXTRACTOR_PROMPT_SHORT.format(webpage_content=result, goal=goal)}]
                            raw = summary_page_func(messages, max_retries=max_retries)
                            raw_json = json.loads(raw)
                            break
                        except Exception:
                            pass
                    
                    # Retry by fetch the summary content only
                    try:
                        start_marker = '"summary": "'
                        start_index = raw.find(start_marker)
                        if start_index != -1:
                            content_start_index = start_index + len(start_marker)
                            last_brace_index = raw.rfind("}")
                            if last_brace_index > content_start_index:
                                end_index = raw.rfind('"', content_start_index, last_brace_index)
                                if end_index != -1:
                                    summary_content = raw[content_start_index:end_index]
                                    raw_json = {
                                        "evidence": "No evidence content available, please refer to the summary.",
                                        "summary": summary_content,
                                    }
                                    break
                    except Exception:
                        pass
                    parse_retry_times += 1
            
            end_time = time.time()
            print(f"[visit] Readpage Jina time taken: {end_time - start_time} seconds. url: {url}")


            if not raw_json:
                print(f"[visit] Could not parse JSON response for url: {url} ======> Raw response: {raw} || Message: {messages}")
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \nThe provided webpage content could not be accessed. Please check the URL or file format.\n\n"
                useful_information += "Summary: \nThe webpage content could not be processed, and therefore, no information is available.\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw_json.get("evidence", "")) + "\n\n"
                useful_information += "Summary: \n" + str(raw_json.get("summary", "")) + "\n\n"

            return useful_information

        useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
        useful_information += "Evidence in page: \nThe provided webpage content could not be accessed. Please check the URL or file format.\n\n"
        useful_information += "Summary: \nThe webpage content could not be processed, and therefore, no information is available.\n\n"

        return useful_information
