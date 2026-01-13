"""
session_resource This module provides simpler types to use with the server for managing prompts
and tools.
"""

import abc
from abc import abstractmethod


class SessionResource(metaclass=abc.ABCMeta):

    @abstractmethod
    async def destroy(self) -> None:
        pass


# Abstract base class
class SessionResourceManager(metaclass=abc.ABCMeta):
    """Manages session resources for MCP server sessions. abstract class"""

    @abstractmethod
    async def create(self, session_id: str) -> None:
        pass

    @abstractmethod
    async def get_resource(self, session_id: str) -> SessionResource:
        pass

    @abstractmethod
    async def destroy(self, session_id: str) -> None:
        pass
