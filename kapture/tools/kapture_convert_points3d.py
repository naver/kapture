#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
This script converts the points3d file of a kapture between the text (points3d.txt) and PLY (points3d.ply) formats.
"""

from typing import Literal

import numpy as np

from kapture.algo.convert_points3d import kapture_convert_points3d, logger
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


def convert_points3d(input: str,
                     format: Literal['ply', 'txt'] = 'ply',
                     keep_source: bool = False,
                     float32: bool = False,
                     timestamp_encoding: Literal['relative', 'absolute'] = 'relative',
                     force: bool = False,
                     verbose: Verbosity = DEFAULT_VERBOSITY,
                     quiet: bool = False) -> None:
    """
    Convert in place the points3d of a kapture to reconstruction/points3d.ply or reconstruction/points3d.txt.

    The point order (the 3d point ids of the observations) is kept. The source is the file of the other format, or
    the file of the requested format itself (to re-encode a PLY). The source is removed once converted.

    :param input: path to the top directory of the kapture
    :param format: format to convert to
    :param keep_source: do not remove the source file
    :param float32: PLY: store x, y, z as float32 instead of float64 (half the size, ~0.1 mm at 1 km)
    :param timestamp_encoding: PLY: how per point timestamps are stored in the double property: relative to the
                               first one, given in the header (exact), or absolute (truncated to 53 bits)
    :param force: overwrite the existing file of the requested format
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    kapture_convert_points3d(input, format,
                             keep_source=keep_source,
                             xyz_dtype=np.float32 if float32 else np.float64,
                             timestamp_encoding=timestamp_encoding,
                             force=force)


def main():
    run_cli(convert_points3d)


if __name__ == '__main__':
    main()
