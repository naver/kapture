#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script merges several kapture dataset to a new kapture. We might also optionally remove some of the data by type.
"""

import argparse
import logging
import os
# kapture
import path_to_kapture
import kapture
import kapture.utils.logging
from typing import List

from kapture.algo.merge_keep_ids import merge_keep_ids
from kapture.algo.merge_remap import merge_remap
from kapture.io.csv import kapture_from_dir, kapture_to_dir
from kapture.io.records import get_image_fullpath
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.records import TransferAction


logger = logging.getLogger('merge')


def merge_kaptures(kapture_path_list: List[str],
                   merged_path: str,
                   keep_sensor_ids:  bool,
                   images_import_strategy: TransferAction = TransferAction.skip,
                   skip: List[str] = [],
                   force: bool = False) -> None:
    """
    Merge a list of kapture dataset to a new one.

    :param kapture_path_list: list of path to the top directory of the kapture datasets to merge
    :param merged_path: path to the merged top directory kapture to create
    :param keep_sensor_ids: if True, will keep the original sensor identifiers. Otherwise, might rename them.
    :param skip: list of kapture data type names to optionally skip (trajectories, records_camera, descriptors, ...)
    :param force: If True, silently overwrite kapture files if already exists.
    """
    os.makedirs(merged_path, exist_ok=True)
    delete_existing_kapture_files(merged_path, force_erase=force)

    skip_list = []
    if 'trajectories' in skip:
        skip_list.append(kapture.Trajectories)
    if 'records_camera' in skip:
        skip_list.append(kapture.RecordsCamera)
    if 'records_lidar' in skip:
        skip_list.append(kapture.RecordsLidar)
    if 'records_wifi' in skip:
        skip_list.append(kapture.RecordsWifi)
    if 'records_gnss' in skip:
        skip_list.append(kapture.RecordsGnss)
    if 'keypoints' in skip:
        skip_list.append(kapture.Keypoints)
    if 'descriptors' in skip:
        skip_list.append(kapture.Descriptors)
    if 'global_features' in skip:
        skip_list.append(kapture.GlobalFeatures)
    if 'matches' in skip:
        skip_list.append(kapture.Matches)
    if 'points3d' in skip:
        skip_list.append(kapture.Points3d)
    if 'observations' in skip:
        skip_list.append(kapture.Observations)

    kapture_data_list = []
    for kapture_path in kapture_path_list:
        logger.info(f'Loading {kapture_path}')
        kapture_data = kapture_from_dir(kapture_path)
        kapture_data_list.append(kapture_data)

    if keep_sensor_ids:
        merged_kapture = merge_keep_ids(kapture_data_list, skip_list, kapture_path_list,
                                        merged_path, images_import_strategy)
    else:
        merged_kapture = merge_remap(kapture_data_list, skip_list, kapture_path_list,
                                     merged_path, images_import_strategy)

    logger.info('Writing merged kapture data...')
    kapture_to_dir(merged_path, merged_kapture)


def merge_command_line() -> None:
    """
    Do the merge_kaptures of several kapture datasets using the command line parameters provided by the user.
    """
    parser = argparse.ArgumentParser(
        description=(f'merge multiple data in the kapture pivot format. Warning: the folder {get_image_fullpath("")} '
                     'will not be copied, remember to manually copy/create a symlink to it.'
                     ' Warning n°2: the inputs records should be disjoint.'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='input paths to kapture data')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory for the merged data')
    parser.add_argument('--keep-sensor-ids', action='store_true', default=False,
                        help='do not remap sensor ids. if duplicates are found, the first one will be kept')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.skip,
                        help=f'How to import images [skip], '
                        f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output directory if already exists.')
    parser.add_argument('-s', '--skip',
                        choices=['trajectories', 'records_camera', 'records_lidar', 'records_wifi',
                                 'keypoints', 'descriptors', 'global_features',
                                 'matches', 'points3d', 'observations'],
                        nargs='+', default=[], help='data to skip')

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    merge_kaptures(args.inputs, args.output, args.keep_sensor_ids, args.image_transfer, args.skip, args.force)


if __name__ == '__main__':
    merge_command_line()
