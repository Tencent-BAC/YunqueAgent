from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

TOOL_REGISTRY = {}


def register_tool(name, allow_overwrite=False):
    """
    Args:
        name: Tool name
        allow_overwrite: Whether to allow overwriting existing tools
    """
    def decorator(cls):
        if name in TOOL_REGISTRY:
            if allow_overwrite:
                print(f'[Warning] Tool `{name}` already exists! Overwriting with class {cls}.')
            else:
                raise ValueError(f'Tool `{name}` already exists! Please ensure that the tool name is unique.')
        if cls.name and (cls.name != name):
            raise ValueError(f'{cls.__name__}.name="{cls.name}" conflicts with @register_tool(name="{name}").')
        cls.name = name
        TOOL_REGISTRY[name] = cls
        return cls
    return decorator


class BaseTool(ABC):
    """
    Base tool class
    
    Attributes:
        name: Tool name
        description: Tool description
        parameters: Tool parameter definition (JSON Schema format)
    """
    name: str = ''
    description: str = ''
    parameters: Union[List[dict], dict] = []

    def __init__(self, cfg: Optional[dict] = None):

        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(
                f'You must set {self.__class__.__name__}.name, either by @register_tool(name=...) '
                f'or explicitly setting {self.__class__.__name__}.name'
            )

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        """
        Tool call interface
        """
        raise NotImplementedError


class Message:
    def __init__(self, role: str, content: str, name: Optional[str] = None, **kwargs):
        self.role = role
        self.content = content
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Message(role='{self.role}', content='{self.content[:50]}...')"
    
    def __str__(self):
        return self.__repr__()


# Message role constants
ASSISTANT = 'assistant'
USER = 'user'
SYSTEM = 'system'
FUNCTION = 'function'
ROLE = 'role'

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

# Configuration constants
DEFAULT_WORKSPACE = "./workspace"
DEFAULT_MAX_INPUT_TOKENS = 30000


def count_tokens(text: str) -> int:
    """Token counting (rough estimate: Chinese by characters, English by words)"""
    import re
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'\b\w+\b', text))
    return chinese_chars + english_words


def has_chinese_chars(text_list) -> bool:
    """Check if text list contains Chinese characters"""
    import re
    for text in text_list:
        text_str = str(text)
        if re.search(r'[\u4e00-\u9fff]', text_str):
            return True
    return False


def extract_code(text: str) -> str:
    """Extract code blocks from text"""
    import re
    # Try to match code blocks wrapped in triple backticks
    triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
    if triple_match:
        return triple_match.group(1).strip()
    
    # Try to match <code> tags
    code_match = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no special markers, return original text
    return text.strip()


# Tokenizer
class SimpleTokenizer:
    def __call__(self, text: str, **kwargs):
        # Simple tokenization: split by spaces and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return {"input_ids": [tokens]}


tokenizer = SimpleTokenizer()


# Logger
class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")
    
    def debug(self, msg):
        print(f"[DEBUG] {msg}")


logger = Logger()


__all__ = [
    'BaseTool',
    'register_tool',
    'TOOL_REGISTRY',
    'Message',
    'ASSISTANT',
    'USER',
    'SYSTEM',
    'FUNCTION',
    'ROLE',
    'DEFAULT_SYSTEM_MESSAGE',
    'DEFAULT_WORKSPACE',
    'DEFAULT_MAX_INPUT_TOKENS',
    'count_tokens',
    'has_chinese_chars',
    'extract_code',
    'tokenizer',
    'logger',
]
