import asyncio
import logging
import time

from aiground.framework.thirdparty.openmanus.app.agent.manus import Manus
from aiground.framework.thirdparty.openmanus.app.flow.flow_factory import (
    FlowFactory,
    FlowType,
)

LOGGER = logging.getLogger(__name__)


async def run_flow():
    agents = {
        "manus": Manus(),
    }

    try:
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            LOGGER.warning("Empty prompt provided.")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        LOGGER.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            LOGGER.info(f"Request processed in {elapsed_time:.2f} seconds")
            LOGGER.info(result)
        except asyncio.TimeoutError:
            LOGGER.error("Request processing timed out after 1 hour")
            LOGGER.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        LOGGER.info("Operation cancelled by user.")
    except Exception as e:
        LOGGER.error(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())
