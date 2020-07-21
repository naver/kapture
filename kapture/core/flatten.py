# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from typing import Tuple, Dict, List, Set, Any, Iterable, Union


def flatten(element: Union[Any, Dict[Any, Any], List[Any], Set[Any]],
            is_sorted: bool = False) -> Iterable[Tuple[Any, ...]]:
    """
    flatten any nested dictionary into a list of tuples:
    {k1: {l1: v1, l2: v2}, k2: {l3: v3}} -> [(k1, l1, v1), (k1, l2, v2), (k2, l3, v3)]
    or
    {k1: [l1, l2], k2: [v1, v2]} -> [(k1, l1), (k1, l2), (k2, v1), (k2, v2)]
    """
    def sort_func(a): return sorted(a) if is_sorted else a
    if isinstance(element, dict):
        for key, value in sort_func(element.items()):
            for tup in flatten(value, is_sorted):
                yield (key,) + tup
    elif isinstance(element, list) or isinstance(element, set):
        for value in sort_func(element):
            for tup in flatten(value, is_sorted):
                yield tup
    else:
        yield element,
