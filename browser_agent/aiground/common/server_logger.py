# -*- coding: utf-8 -*-
"""
server_logger

Logging setup helpers.
"""
import logging
import logging.handlers


def init_logger(logger_config: dict):
    if logger_config is None:
        logger_config = {}
    logger_filename = logger_config.get("logger_filename", "workspace/business.log")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        logger_filename, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)s %(message)s"
    )
    logger.handlers = []
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    enable_stdout = logger_config.get("enable_stdout", False)
    if enable_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
