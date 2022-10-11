#!/usr/bin/env python3
# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
This script imports dsacstar results.

"""

import argparse
import logging
import os
import os.path as path
import quaternion
import numpy as np

# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_to_dir
from kapture.io.structure import delete_existing_kapture_files

logger = logging.getLogger('import_image_list_with_poses')


def import_dsacstar_results(images_list_file_path: str,
                                 kapture_path: str,
                                 force_overwrite_existing: bool = False) -> None:
    """
    Imports the list of images with their poses to a kapture.

    :param images_list_file_path: file containing the list of images with their poses
    :param kapture_path: path to kapture root directory.
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    """
    if not path.exists(images_list_file_path):
        raise FileNotFoundError(images_list_file_path)
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()
    cameras = kapture.Sensors()
    camera_id = 'not_provided'
    cameras[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, [0, 0], camera_id)

    logger.info(f'Loading data from {images_list_file_path}')

    # Parse image list data file
    with open(images_list_file_path, 'rt') as file:
        images_list = file.readlines()
        images_list = [line.rstrip() for line in images_list if line != '\n']
        for i in range(0, len(images_list)):
            image_line = images_list[i].split()
            image_path, qw, qx, qy, qz, tx, ty, tz = image_line[0:8]
            timestamp = i
            images[(int(timestamp), camera_id)] = image_path
            # Create pose
            rotation = None
            translation = None
            if qw != '' and qx != '' and qy != '' and qz != '':
                rotation = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])
            if tx != '' and ty != '' and tz != '':
                translation = np.array([[float(tx)], [float(ty)], [float(tz)]], dtype=np.float)
            pose = kapture.PoseTransform(rotation, translation)
            trajectories[(int(timestamp), camera_id)] = pose

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_path, imported_kapture)


def import_dsacstar_results_command_line() -> None:
    """
    import dsacstar results
    """
    parser = argparse.ArgumentParser(description='import images with their poses (trajectory+images)'
                                                 ' to the kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-l', '--images_list_file', required=True,
                        help='path to images (with their poses) list file')
    parser.add_argument('-k', '--kapture', required=True, help='kapture directory')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_dsacstar_results(args.images_list_file,
                                 args.kapture, args.force)


if __name__ == '__main__':
    import_dsacstar_results_command_line()
