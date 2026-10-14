#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports images from a folder in the kapture format
"""

import logging
import os
import os.path as path
import PIL
from PIL import Image

# kapture
import kapture
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir, sensors_from_file
from kapture.utils.paths import path_secure
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('image_folder')


def import_image_folder(
        input: str,
        output: str,
        force: bool = False,
        image_transfer: TransferAction = TransferAction.link_absolute,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False
) -> None:
    """
    Imports the images of a folder to a kapture. This creates only images and cameras.

    :param input: path to directory containing the images.
    :param output: path to kapture root directory.
    :param force: silently overwrite kapture files if already exists.
    :param image_transfer: choose how to import actual image files.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    os.makedirs(output, exist_ok=True)
    delete_existing_kapture_files(output, force_erase=force)

    images = kapture.RecordsCamera()

    file_list = [path.relpath(path.join(dirpath, filename), input)
                 for dirpath, dirs, filenames in os.walk(input)
                 for filename in filenames]
    file_list = sorted(file_list)

    try:
        sensors_filepath = path.join(input, 'sensors.txt')
        sensors = sensors_from_file(sensors_filepath)
        sensors_given = True
    except OSError:
        logger.info('image folder has no extra sensor info')
        sensors = kapture.Sensors()
        sensors_given = False

    logger.info('starting conversion...')
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with Image.open(path.join(input, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            logger.info(f'Skipping invalid image file {filename}')
            continue

        if sensors_given:
            if len(sensors) == 1:
                # in case of single camera, just take the name from sensors.txt
                camera_id = list(sensors.keys())[0]
            else:
                # guess sensor_id from dirname
                camera_id = path.dirname(path.join(input, filename)).split(os.sep)[-1]

            if camera_id not in sensors:
                logger.critical(f'camera {camera_id} is not found in sensors.txt')
                raise KeyError(f'camera {camera_id} is not found in sensors.txt')

        else:
            camera_id = f'sensor{n}'
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)

        images[(n, camera_id)] = path_secure(filename)  # don't forget windows

    # import (copy) image files.
    logger.info('import image files ...')
    filename_list = [f for _, _, f in kapture.flatten(images)]
    import_record_data_from_dir_auto(input, output, filename_list, image_transfer)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=sensors, records_camera=images)
    logger.info('writing imported data...')
    kapture_to_dir(output, imported_kapture)


def main():
    run_cli(import_image_folder)


if __name__ == '__main__':
    main()
