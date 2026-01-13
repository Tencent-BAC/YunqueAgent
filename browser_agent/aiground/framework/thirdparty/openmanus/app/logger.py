import logging

LOGGER = logging.getLogger("openmanus")


def test():
    LOGGER.info("Starting application")
    LOGGER.debug("Debug message")
    LOGGER.warning("Warning message")
    LOGGER.error("Error message")
    LOGGER.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        LOGGER.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    test()
