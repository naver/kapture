#!/usr/bin/env python3
# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
This script imports a trajectory file with images in the kapture format
The trajectory file contains the following info on every line: timestamp, camera id, image path, 6d pose

"""

import argparse
import logging
import os
import os.path as path
import PIL
from PIL import Image
import quaternion
import numpy as np
from typing import List

# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import table_from_file, kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('import_localized_images')


def import_localized_images(localized_file_path: str,
                            images_dir_path: str,
                            kapture_path: str,
                            force_overwrite_existing: bool = False,
                            images_import_method: TransferAction = TransferAction.skip,
                            do_not_import_images: bool = False) -> None:
    """
    Imports the list of images to a kapture. This creates only images and cameras.

    :param localized_file_path: file containing the list of localized images with their poses
    :param images_dir_path: top directory of the images path. If not defined, the images path in the localized file
        must be full path.
    :param kapture_path: path to kapture root directory.
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    :param do_not_import_images: when true, do not import the image files.
    """
    if not path.exists(localized_file_path):
        raise FileNotFoundError(localized_file_path)
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()
    filename_list: List[str] = []
    image_name_list: List[str] = []

    logger.info(f'Loading data from {localized_file_path}')
    # Find the top directory of all image files
    with open(localized_file_path, 'rt') as file:
        localized_data = table_from_file(file)
        for localized_line in localized_data:
            image_path = localized_line[2]
            if not do_not_import_images and not images_dir_path and not path.isabs(image_path):
                raise ValueError(f'Can not import a relative image file without top directory: {image_path}')
            if not path.isabs(image_path) and images_dir_path:
                image_path = path.join(images_dir_path, image_path)
            filename_list.append(image_path)
    common_images_path = path.commonpath(filename_list)

    # Parse localized data file
    with open(localized_file_path, 'rt') as file:
        localized_data = table_from_file(file)
        for localized_line in localized_data:
            timestamp, camera_id, image_path, qw, qx, qy, qz, tx, ty, tz = localized_line
            # Add top directory if necessary
            if not path.isabs(image_path) and images_dir_path:
                image_path = path.join(images_dir_path, image_path)
            if path.exists(image_path):
                # Create corresponding camera model
                model = kapture.CameraType.UNKNOWN_CAMERA.value
                try:
                    # lazy open
                    with Image.open(image_path) as im:
                        width, height = im.size
                        model_params = [width, height]
                        cameras[camera_id] = kapture.Camera(model, model_params, camera_id)
                except (OSError, PIL.UnidentifiedImageError):
                    # It is not a valid image: skip it
                    logger.info(f'Skipping invalid image file {image_path}')
                if not do_not_import_images:
                    image_name = path.relpath(image_path, common_images_path)
                    image_name_list.append(image_name)
                    images[(int(timestamp), camera_id)] = image_name
            else:
                logger.debug(f'Missing image file {image_path}')
            # Create pose
            rotation = None
            translation = None
            if qw != '' and qx != '' and qy != '' and qz != '':
                rotation = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])
            if tx != '' and ty != '' and tz != '':
                translation = np.array([[float(tx)], [float(ty)], [float(tz)]], dtype=np.float)
            pose = kapture.PoseTransform(rotation, translation)
            trajectories[(int(timestamp), camera_id)] = pose

    # import (copy) image files.
    if not do_not_import_images:
        logger.info('import image files ...')
        import_record_data_from_dir_auto(common_images_path, kapture_path, image_name_list, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_path, imported_kapture)


def import_localized_images_command_line() -> None:
    """
    Do the localized images import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='import localized images (trajectory+images) to the kapture format.')
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
    parser.add_argument('-l', '--localized_file', required=True,
                        help='path to localized images list file')
    parser.add_argument('--top_dir_images', default='',
                        help='path to top directory of images.'
                             ' Necessary only if the images path are relative in the localized file')
    parser.add_argument('-k', '--kapture', required=True, help='kapture directory')
    parser.add_argument('--image_action', type=TransferAction, default=TransferAction.copy,
                        help=f'How to import images [copy], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('--do_not_import_images', action='store_true', default=False,
                        help='Do not import the images files, but their poses (the trajectory) only.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_localized_images(args.localized_file, args.top_dir_images,
                            args.kapture, args.force,
                            args.image_action, args.do_not_import_images)


if __name__ == '__main__':
    import_localized_images_command_line()
