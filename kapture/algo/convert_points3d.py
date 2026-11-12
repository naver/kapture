# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Conversion of the points3d file of a kapture between its text (points3d.txt) and PLY (points3d.ply) formats.
"""

import logging
import os
import os.path as path
from typing import Optional

import numpy as np

from kapture.io.csv import POINTS3D_FORMATS, get_points3d_fullpath, points3d_from_file, points3d_to_file

logger = logging.getLogger('convert_points3d')


def kapture_convert_points3d(kapture_dirpath: str,
                             points3d_format: str,
                             keep_source: bool = False,
                             xyz_dtype: type = np.float64,
                             timestamp_encoding: str = 'relative',
                             force: bool = False) -> Optional[str]:
    """
    Converts in place the points3d file of a kapture to the given format. The point order (the 3d point ids used by
    the observations) is kept.
    The source is the points3d file of the other format if any, else the one of the given format (eg. to change the
    PLY encoding of the coordinates or timestamps).

    :param kapture_dirpath: kapture directory root path
    :param points3d_format: 'ply' or 'txt'
    :param keep_source: do not remove the source file once converted
    :param xyz_dtype: PLY only: np.float64 (default, no loss) or np.float32
    :param timestamp_encoding: PLY only: 'relative' (default, exact) or 'absolute' (truncated to double precision)
    :param force: overwrite the existing file of the given format
    :return: path of the converted file, or None if the kapture has no points3d.
    """
    if points3d_format not in POINTS3D_FORMATS:
        raise ValueError(f'unknown points3d format {points3d_format}, expected one of {POINTS3D_FORMATS}')
    other_format = next(other for other in POINTS3D_FORMATS if other != points3d_format)
    target_filepath = get_points3d_fullpath(kapture_dirpath, points3d_format)
    source_filepath = get_points3d_fullpath(kapture_dirpath, other_format)
    if not path.isfile(source_filepath):
        source_filepath = target_filepath
    if not path.isfile(source_filepath):
        logger.warning(f'no points3d in {kapture_dirpath}')
        return None
    if source_filepath != target_filepath and path.isfile(target_filepath) and not force:
        raise FileExistsError(f'{target_filepath} already exists, use force to overwrite it')

    logger.info(f'loading {source_filepath}')
    points3d = points3d_from_file(source_filepath)
    logger.info(f'writing {len(points3d)} points to {target_filepath}')
    points3d_to_file(target_filepath, points3d, xyz_dtype=xyz_dtype, timestamp_encoding=timestamp_encoding)
    if source_filepath != target_filepath and not keep_source:
        logger.info(f'removing {source_filepath}')
        os.remove(source_filepath)
    return target_filepath
