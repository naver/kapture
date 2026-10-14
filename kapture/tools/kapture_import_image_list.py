#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports a list of images in the kapture format
"""

import logging
import os
import os.path as path
import PIL
from PIL import Image
from typing import List

# kapture
import kapture
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('import_image_list')


def import_image_list(inputs: List[str],
                      output: str,
                      images: str = '',
                      force: bool = False,
                      image_transfer: TransferAction = TransferAction.link_absolute,
                      verbose: Verbosity = DEFAULT_VERBOSITY,
                      quiet: bool = False) -> None:
    """
    Imports the list of images to a kapture. This creates only images and cameras.
    Each line of an image list file is either "image_name camera_type camera_params" or just "image_name".

    :param inputs: paths to the image list text files
    :param output: path to kapture root directory.
    :param images: path to images directory. Only needed when inputs do not have camera parameters.
    :param force: silently overwrite kapture files if already exists.
    :param image_transfer: choose how to import actual image files.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)

    assert isinstance(inputs, list)
    os.makedirs(output, exist_ok=True)
    delete_existing_kapture_files(output, force_erase=force)

    cameras = kapture.Sensors()
    records_camera = kapture.RecordsCamera()

    offset = 0
    logger.info('starting conversion...')
    for images_list_filename in inputs:
        logger.info(f'loading {images_list_filename}')
        with open(images_list_filename) as file:
            images_list = file.readlines()
            # remove end line char and empty lines
            images_list = [line.rstrip() for line in images_list if line != '\n']

            for i in range(0, len(images_list)):
                line = images_list[i].split()
                image_file_name = line[0]
                if len(line) > 1:
                    model = line[1]
                    model_params = line[2:]
                else:
                    model = kapture.CameraType.UNKNOWN_CAMERA.value
                    try:
                        # lazy open
                        with Image.open(path.join(images, image_file_name)) as im:
                            width, height = im.size
                            model_params = [width, height]
                    except (OSError, PIL.UnidentifiedImageError):
                        # It is not a valid image: skip it
                        logger.info(f'Skipping invalid image file {image_file_name}')
                        continue

                camera_id = f'sensor{i + offset}'
                cameras[camera_id] = kapture.Camera(model, model_params)
                records_camera[(i + offset, camera_id)] = image_file_name
            offset += len(images_list)

    # import (copy) image files.
    logger.info('import image files ...')
    filename_list = [f for _, _, f in kapture.flatten(records_camera)]
    import_record_data_from_dir_auto(images, output, filename_list, image_transfer)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=records_camera)
    logger.info('writing imported data...')
    kapture_to_dir(output, imported_kapture)


def main():
    run_cli(import_image_list, aliases={'--images': ['-im', '--image_path']})


if __name__ == '__main__':
    main()
