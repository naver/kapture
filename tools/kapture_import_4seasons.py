#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a 4seasons model into a kapture.
The data structure is defined here:
    https://www.4seasons-dataset.com/documentation

The 4Seasons dataset contains recordings from a stereo-inertial camera system coupled with a high-end RTK-GNSS.

For each sequence, the recorded data is stored in the following structure:
├── KeyFrameData
├── distorted_images
│   ├── cam0
│   └── cam1
├── undistorted_images
│   ├── cam0
│   └── cam1
├── GNSSPoses.txt
├── Transformations.txt
├── imu.txt
├── result.txt
├── septentrio.nmea
└── times.txt
"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from PIL import Image
from tqdm import tqdm
# kapture
import path_to_kapture  # noqa: F401
from kapture.core.Sensors import SENSOR_TYPE_DEPTH_CAM
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from typing import List

logger = logging.getLogger('4seasons')


def _import_4seasons_sequence(
        sequence_root: str,
        sequence_list: List[str],
        kapture_dir_path: str,
        images_import_method: TransferAction,
        force_overwrite_existing: bool):
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    snapshots = kapture.RecordsCamera()
    depth_maps = kapture.RecordsDepth()
    trajectories = kapture.Trajectories()
    rigs = kapture.Rigs()
    sensors = kapture.Sensors()


def import_4seasons(
        d4seasons_path: str,
        kapture_dir_path: str,
        force_overwrite_existing: bool = False,
        images_import_method: TransferAction = TransferAction.skip
) -> None:
    """
    Imports 4seasons dataset and save them as kapture.
    :param d4seasons_path: path to the 4seasons root path
    :param kapture_dir_path: path to kapture output top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files
    """
    os.makedirs(kapture_dir_path, exist_ok=True)


def import_4seasons_command_line() -> None:
    """
    Imports 4seasons dataset and save them as kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='Imports 4seasons dataset files to the kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i', '--input', required=True, help='input path to RIO10 root path')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_4seasons(d4seasons_path=args.input,
                    kapture_dir_path=args.output,
                    force_overwrite_existing=args.force,
                    images_import_method=args.image_transfer)


if __name__ == '__main__':
    import_4seasons_command_line()
