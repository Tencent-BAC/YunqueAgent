import asyncio
import logging

from aiground.framework.thirdparty.openmanus.app.agent.manus import Manus

LOGGER = logging.getLogger(__name__)


async def main():
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        prompt = input("Enter your prompt: ")
        if not prompt.strip():
            LOGGER.warning("Empty prompt provided.")
            return

        LOGGER.warning("Processing your request...")
        await agent.run(prompt)
        LOGGER.info("Request processing completed.")
    except KeyboardInterrupt:
        LOGGER.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
