# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Additional collections and base python objects.
"""

import collections
from enum import Enum


class AutoEnum(Enum):
    """
    When the auto operator is used, value = name
    see https://docs.python.org/3/library/enum.html#using-automatic-values
    """
    def _generate_next_value_(name, start, count, last_values):
        return name


class LimitedDictionary(collections.OrderedDict):
    """
    An ordered dictionary with a limited size, evicting the oldest inserted when full
    """

    def __init__(self, maxsize=64, *args, **kwds):
        self._maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            oldest = next(iter(self))
            del self[oldest]
