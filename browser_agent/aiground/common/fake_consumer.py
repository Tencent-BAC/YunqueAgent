# -*- coding: utf-8 -*-
"""
fake_consumer

An empty stub implementation to allow local runs on macOS.
"""

import logging

LOGGER = logging.getLogger(__name__)

LOGGER.warning("fake_consumer is not supported on macos, use it only for testing")


def create_consumer_by_default_config_file():
    pass


class Instance:
    def __init__(self):
        pass
