# -*- coding: utf-8 -*-
"""
singleton

Singleton decorator.
"""
import os


def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton():
        key = (str(cls), str(os.getpid()))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]

    return _singleton
