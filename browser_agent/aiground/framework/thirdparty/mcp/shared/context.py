from dataclasses import dataclass
from typing import Any, Generic

from typing_extensions import TypeVar

from aiground.framework.thirdparty.mcp.shared.session import BaseSession
from aiground.framework.thirdparty.mcp.types import RequestId, RequestParams

SessionT = TypeVar("SessionT", bound=BaseSession[Any, Any, Any, Any, Any])
LifespanContextT = TypeVar("LifespanContextT")
RequestT = TypeVar("RequestT", default=Any)


@dataclass
class RequestContext(Generic[SessionT, LifespanContextT, RequestT]):
    request_id: RequestId
    meta: RequestParams.Meta | None
    session: SessionT
    lifespan_context: LifespanContextT
    request: RequestT | None = None
