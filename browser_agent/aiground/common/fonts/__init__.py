# -*- coding: utf-8 -*-
"""
fonts

Font utilities.
"""
import os


def get_base_path():
    """Return the fonts directory path."""
    return os.path.abspath(
        os.path.dirname(os.path.realpath(__file__)),
    )


def wqy_microhei_path():
    base_path = get_base_path()
    return os.path.join(base_path, "wqy-microhei.ttc")
