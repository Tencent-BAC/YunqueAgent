# -*- coding: utf-8 -*-
"""
message_queue

Minimal message queue utility.
"""

import queue


class MessageQueue(object):
    def __init__(self):
        self._queue = queue.Queue()
        pass

    def produce(self, message: str):
        self._queue.put(message)

    def consume(self):
        return self._queue.get()
