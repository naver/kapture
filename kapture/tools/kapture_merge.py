#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script merges several kapture dataset to a new kapture. We might also optionally remove some of the data by type.
"""

import logging
import os
# kapture
import kapture
from typing import List, Literal

from kapture.algo.merge_keep_ids import merge_keep_ids
from kapture.algo.merge_remap import merge_remap
from kapture.io.csv import get_all_tar_handlers, kapture_from_dir, kapture_to_dir
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.records import TransferAction
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('merge')

SkipType = Literal['trajectories', 'records_camera', 'records_depth', 'records_lidar', 'records_wifi',
                   'records_bluetooth', 'records_gnss', 'records_accelerometer', 'records_gyroscope',
                   'records_magnetic', 'keypoints', 'descriptors', 'global_features',
                   'matches', 'points3d', 'observations']


def merge_kaptures(inputs: List[str],  # noqa: C901: function really not complex
                   output: str,
                   keep_sensor_ids: bool = False,
                   image_transfer: TransferAction = TransferAction.skip,
                   skip: List[SkipType] = [],
                   force: bool = False,
                   verbose: Verbosity = DEFAULT_VERBOSITY,
                   quiet: bool = False) -> None:
    """
    Merge several kapture datasets into a new one. Warning: the images folder is not copied by default
    (image_transfer=skip), remember to manually copy/create a symlink to it. The inputs records should be disjoint.

    :param inputs: paths to the top directories of the kapture datasets to merge
    :param output: path to the merged top directory kapture to create
    :param keep_sensor_ids: do not remap sensor ids. If duplicates are found, the first one will be kept.
    :param image_transfer: how to import image files
    :param skip: kapture data types to skip
    :param force: silently overwrite kapture files if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    os.makedirs(output, exist_ok=True)
    delete_existing_kapture_files(output, force_erase=force)

    skip_list = []
    if 'trajectories' in skip:
        skip_list.append(kapture.Trajectories)
    if 'records_camera' in skip:
        skip_list.append(kapture.RecordsCamera)
    if 'records_depth' in skip:
        skip_list.append(kapture.RecordsDepth)
    if 'records_lidar' in skip:
        skip_list.append(kapture.RecordsLidar)
    if 'records_wifi' in skip:
        skip_list.append(kapture.RecordsWifi)
    if 'records_bluetooth' in skip:
        skip_list.append(kapture.RecordsBluetooth)
    if 'records_gnss' in skip:
        skip_list.append(kapture.RecordsGnss)
    if 'records_accelerometer' in skip:
        skip_list.append(kapture.RecordsAccelerometer)
    if 'records_gyroscope' in skip:
        skip_list.append(kapture.RecordsGyroscope)
    if 'records_magnetic' in skip:
        skip_list.append(kapture.RecordsMagnetic)
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
    kapture_tarcollection_list = []
    try:
        for kapture_path in inputs:
            logger.info(f'Loading {kapture_path}')
            tar_handlers = get_all_tar_handlers(kapture_path)
            kapture_data = kapture_from_dir(kapture_path, tar_handlers=tar_handlers, skip_list=skip_list)
            kapture_data_list.append(kapture_data)
            kapture_tarcollection_list.append(tar_handlers)

        if keep_sensor_ids:
            merged_kapture = merge_keep_ids(kapture_data_list, skip_list,
                                            inputs, kapture_tarcollection_list,
                                            output, image_transfer)
        else:
            merged_kapture = merge_remap(kapture_data_list, skip_list,
                                         inputs, kapture_tarcollection_list,
                                         output, image_transfer)
    finally:
        for tar_handlers in kapture_tarcollection_list:
            tar_handlers.close()

    logger.info('Writing merged kapture data...')
    kapture_to_dir(output, merged_kapture)


def main():
    run_cli(merge_kaptures, aliases={'--skip': ['-s']})


if __name__ == '__main__':
    main()
