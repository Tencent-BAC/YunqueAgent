from enum import Enum
from typing import Dict, List, Union

from aiground.framework.thirdparty.openmanus.app.agent.base import BaseAgent
from aiground.framework.thirdparty.openmanus.app.flow.base import BaseFlow
from aiground.framework.thirdparty.openmanus.app.flow.planning import PlanningFlow


class FlowType(str, Enum):
    PLANNING = "planning"


class FlowFactory:
    """Factory for creating different types of flows with support for multiple agents"""

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        return flow_class(agents, **kwargs)
