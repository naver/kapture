#!/usr/bin/env python3
# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
This script imports a trajectory file with images in the kapture format
The trajectory file contains the following info on every line: timestamp, camera id, image path, 6d pose

"""

import logging
import os
import os.path as path
import PIL
from PIL import Image
import quaternion
import numpy as np
from typing import List

# kapture
import kapture
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY
from kapture.io.csv import table_from_file, kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.io.structure import delete_existing_kapture_files
from kapture.utils.open_cv import import_opencv_camera_calibration

logger = logging.getLogger('import_image_list_with_poses')


def find_images_files_top_directory(images_list_file_path, images_dir_path, do_not_import_images):
    """
    Find the top directory of all image files
    """
    filename_list: List[str] = []
    with open(images_list_file_path, 'rt') as file:
        images_list_data = table_from_file(file)
        for image_line in images_list_data:
            image_path = image_line[2]
            if not do_not_import_images and not images_dir_path and not path.isabs(image_path):
                raise ValueError(f'Can not import a relative image file without top directory: {image_path}')
            if not path.isabs(image_path) and images_dir_path:
                image_path = path.join(images_dir_path, image_path)
            filename_list.append(image_path)
    common_images_path = path.commonpath(filename_list)
    return common_images_path


def import_image_list_with_poses(input: str,
                                 output: str,
                                 images: str = '',
                                 calibration: str = '',
                                 force: bool = False,
                                 image_transfer: TransferAction = TransferAction.copy,
                                 import_images: bool = True,
                                 verbose: Verbosity = DEFAULT_VERBOSITY,
                                 quiet: bool = False) -> None:
    """
    Imports a list of images with their poses (trajectory+images) to a kapture.

    :param input: path to the file containing the list of images with their poses
    :param output: path to kapture root directory.
    :param images: path to top directory of images. Necessary only if the images path are relative in the list file.
    :param calibration: path to opencv camera calibration YAML file (YAML 1.1).
                        If missing, will compute a simple camera model based on the images.
    :param force: silently overwrite kapture files if already exists.
    :param image_transfer: choose how to import actual image files.
    :param import_images: import the image files. If false, imports their poses (the trajectory) only.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)

    if not path.exists(input):
        raise FileNotFoundError(input)
    os.makedirs(output, exist_ok=True)
    delete_existing_kapture_files(output, force_erase=force)

    cameras = kapture.Sensors()
    camera_model_provided = False
    if calibration:
        if not path.exists(calibration):
            raise FileNotFoundError(calibration)
        camera = import_opencv_camera_calibration(calibration)
        cameras[camera.name] = camera
        camera_model_provided = True
    records_camera = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()
    image_name_list: List[str] = []

    logger.info(f'Loading data from {input}')
    common_images_path = find_images_files_top_directory(input, images, not import_images)

    # Parse image list data file
    with open(input, 'rt') as file:
        images_list_data = table_from_file(file)
        for image_line in images_list_data:
            timestamp, camera_id, image_path, qw, qx, qy, qz, tx, ty, tz = image_line
            # Add top directory if necessary
            if not path.isabs(image_path) and images:
                image_path = path.join(images, image_path)
            if path.exists(image_path):
                if not camera_model_provided:
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
                if import_images:
                    image_name = path.relpath(image_path, common_images_path)
                    image_name_list.append(image_name)
                    records_camera[(int(timestamp), camera_id)] = image_name
            else:
                logger.debug(f'Missing image file {image_path}')
            # Create pose
            rotation = None
            translation = None
            if qw != '' and qx != '' and qy != '' and qz != '':
                rotation = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])
            if tx != '' and ty != '' and tz != '':
                translation = np.array([[float(tx)], [float(ty)], [float(tz)]], dtype=float)
            pose = kapture.PoseTransform(rotation, translation)
            trajectories[(int(timestamp), camera_id)] = pose

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=records_camera, trajectories=trajectories)
    logger.info('writing imported data...')
    kapture_to_dir(output, imported_kapture)

    # import (copy) image files.
    if import_images:
        logger.info('import image files ...')
        import_record_data_from_dir_auto(common_images_path, output, image_name_list, image_transfer)


def main():
    run_cli(import_image_list_with_poses, aliases={
        '--input': ['-l', '--images_list_file'],
        '--output': ['-k', '--kapture'],
        '--images': ['--top_dir_images'],
        '--calibration': ['-c', '--camera_model_file'],
        '--image_transfer': ['--image_action'],
    })


if __name__ == '__main__':
    main()
