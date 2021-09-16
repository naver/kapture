#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports images from a folder in the kapture format
"""

import argparse
import logging
import os
import os.path as path
import PIL
from PIL import Image

# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.utils.paths import path_secure
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('image_folder')


def import_image_folder(images_path: str,
                        kapture_path: str,
                        force_overwrite_existing: bool = False,
                        images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Imports the images of a folder to a kapture. This creates only images and cameras.

    :param images_path: path to directory containing the images.
    :param kapture_path: path to kapture root directory.
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()

    file_list = [path.relpath(path.join(dirpath, filename), images_path)
                 for dirpath, dirs, filenames in os.walk(images_path)
                 for filename in filenames]
    file_list = sorted(file_list)

    sensor_info = {}  # {sensor_id:[sensor_type, sensor_params+]}
    sensor_flag = True
    try:
        sensors = open(path.join(images_path, "sensors.txt"), 'r')
        lines = sensors.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '#':
                continue
            line = line.split(", ")
            sensor_info[line[0]] = line[3:]

    except OSError:
        logger.info('image folder has no extra sensor info')
        sensor_flag = False

    camera_types = {'SIMPLE_PINHOLE': kapture.CameraType.SIMPLE_PINHOLE,
                    'PINHOLE': kapture.CameraType.PINHOLE,
                    'SIMPLE_RADIAL': kapture.CameraType.SIMPLE_RADIAL,
                    'RADIAL': kapture.CameraType.RADIAL,
                    'OPENCV': kapture.CameraType.OPENCV,
                    'OPENCV_FISHEYE': kapture.CameraType.OPENCV_FISHEYE,
                    'FULL_OPENCV': kapture.CameraType.FULL_OPENCV,
                    'FOV': kapture.CameraType.FOV,
                    'SIMPLE_RADIAL_FISHEYE': kapture.CameraType.SIMPLE_RADIAL_FISHEYE,
                    'RADIAL_FISHEYE': kapture.CameraType.RADIAL_FISHEYE,
                    'THIN_PRISM_FISHEYE': kapture.CameraType.THIN_PRISM_FISHEYE,
                    'UNKNOWN_CAMERA': kapture.CameraType.UNKNOWN_CAMERA}

    logger.info('starting conversion...')
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with Image.open(path.join(images_path, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            logger.info(f'Skipping invalid image file {filename}')
            continue

        if sensor_flag:
            camera_id = path.dirname(path.join(images_path, filename)).split(os.sep)[-1]
            images[(n, camera_id)] = path_secure(filename)  # don't forget windows
            try:
                cameras[camera_id] = kapture.Camera(camera_types[sensor_info[camera_id][0]], sensor_info[camera_id][1:])
            except KeyError:
                logger.info(f'{camera_id} has no valid camera type')
                cameras[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)
        else:
            camera_id = f'sensor{n}'
            images[(n, camera_id)] = path_secure(filename)  # don't forget windows
            cameras[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)

    # import (copy) image files.
    logger.info('import image files ...')
    filename_list = [f for _, _, f in kapture.flatten(images)]
    import_record_data_from_dir_auto(images_path, kapture_path, filename_list, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_path, imported_kapture)


def import_image_folder_command_line() -> None:
    """
    Do the image list import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='imports images from a folder in the kapture format')
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
    parser.add_argument('-i', '--input', required=True, help='input path to images root folder')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_image_folder(args.input, args.output, args.force, args.image_transfer)


if __name__ == '__main__':
    import_image_folder_command_line()
