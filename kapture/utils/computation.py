# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
Computation helper operations.
"""


def num_digits(n) -> int:
    """
    Compute the number of digits of a number

    :return: number of digits
    """
    if not isinstance(n, int):
        raise TypeError(f'Invalid n: not an integer {n}')
    count = 1
    while (int(n / 10)) != 0:
        count += 1
        n = int(n / 10)
    return count
