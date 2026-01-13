# -*- coding: utf-8 -*-
"""
dict_args dict args
"""
import copy


def _convert_value(value):
    if isinstance(value, dict):
        return DictArgs(value)
    elif isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    else:
        return value


class DictArgs(dict):
    def __init__(self, data: dict):
        super().__init__()
        # super().__setattr__("_state", {})
        if not isinstance(data, dict):
            raise ValueError("data not a dict")
        for key, value in data.items():
            setattr(self, key, _convert_value(value))

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        # return self.get(key)
        return self[key]

    def __deepcopy__(self, memo):
        new_dict_args = DictArgs({})
        for key, value in self.items():
            new_dict_args[key] = copy.deepcopy(value, memo)
        return new_dict_args

    # Support pickling/serialization
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)
