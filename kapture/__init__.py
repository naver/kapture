# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
All base kapture objects, with their reading and writing functions on disk,
and the conversion from and to other formats.
"""

from .core import *
# silence kapture logging to critical only, except if told otherwise
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
