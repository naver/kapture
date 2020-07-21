# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Logging specific to kapture
"""

import argparse
import logging

logging.basicConfig(format='%(levelname)-8s::%(name)s: %(message)s')


class VerbosityParser(argparse.Action):
    """ accept debug, info, ... or theirs corresponding integer value formatted as string."""

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, (int, str))
        try:  # in case it represent an int, directly get it
            values = int(values)
        except ValueError:  # else ask logging to sort it out
            values = logging.getLevelName(values.upper())
        setattr(namespace, self.dest, values)


def getLogger():
    """
    Get the default kapture logger.

    :return: logger
    """
    return logging.getLogger('kapture')
