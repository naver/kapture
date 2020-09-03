#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports a list of images in the kapture format
"""

import argparse
import logging
import os
import os.path as path
import PIL
from PIL import Image
from typing import List

# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('import_image_list')


def import_image_list(images_list_filenames: List[str],
                      images_dirpath: str,
                      kapture_path: str,
                      force_overwrite_existing: bool = False,
                      images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Imports the list of images to a kapture. This creates only images and cameras.

    :param images_list_filenames: list of text files containing image file names
    :param images_dirpath: path to images directory.
    :param kapture_path: path to kapture root directory.
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """
    assert isinstance(images_list_filenames, list)
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()

    offset = 0
    logger.info('starting conversion...')
    for images_list_filename in images_list_filenames:
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
                        with Image.open(path.join(images_dirpath, image_file_name)) as im:
                            width, height = im.size
                            model_params = [width, height]
                    except (OSError, PIL.UnidentifiedImageError):
                        # It is not a valid image: skip it
                        logger.info(f'Skipping invalid image file {image_file_name}')
                        continue

                camera_id = f'sensor{i + offset}'
                cameras[camera_id] = kapture.Camera(model, model_params)
                images[(i + offset, camera_id)] = image_file_name
            offset += len(images_list)

    # import (copy) image files.
    logger.info('import image files ...')
    filename_list = [f for _, _, f in kapture.flatten(images)]
    import_record_data_from_dir_auto(images_dirpath, kapture_path, filename_list, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_path, imported_kapture)


def import_image_list_command_line() -> None:
    """
    Do the image list import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='import image list text files to the kapture format.')
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
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help=('input path to image list files:\n'
                              'image_name camera_type camera_params\n or\n image_name'))
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('-im', '--image_path', default="",
                        help='path to images. Only needed when inputs do not have camera parameters')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    ####################################################################################################################
    args = parser.parse_args()

    """
    image_name camera_type camera_params
    example: image_name SIMPLE_RADIAL w h f cx cy r
    or
    image_name
    """

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_image_list(
        images_list_filenames=args.inputs,
        images_dirpath=args.image_path,
        kapture_path=args.output,
        force_overwrite_existing=args.force,
        images_import_method=args.image_transfer
    )


if __name__ == '__main__':
    import_image_list_command_line()
