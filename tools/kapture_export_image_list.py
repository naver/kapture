#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Export the list of images of a kapture into a file listing the image files, and optionally their camera parameters.
"""

import logging
import os
import argparse
# kapture
import path_to_kapture
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir
from kapture.utils.paths import safe_remove_file


logger = logging.getLogger('export_image_list')


def export_image_list(kapture_path: str, output_path: str, export_camera_params: bool, force: bool) -> None:
    """
    Export image list in a text file.

    :param kapture_path: top directory of the kapture
    :param output_path: path of the image list file
    :param export_camera_params: if True, add camera parameters after every file name
    :param force: Silently overwrite image list file if already exists.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safe_remove_file(output_path, force)

    kapture_to_export = kapture_from_dir(kapture_path)
    output_content = []
    logger.info('starting conversion...')
    for _, sensor_id, filename in kapture.flatten(kapture_to_export.records_camera, is_sorted=True):
        line = filename
        if export_camera_params:
            camera = kapture_to_export.sensors[sensor_id]
            assert isinstance(camera, kapture.Camera)
            line += ' ' + ' '.join(camera.sensor_params)
        output_content.append(line)

    logger.info('writing exported data...')
    with open(output_path, 'w') as fid:
        fid.write('\n'.join(output_content))


def export_image_list_command_line() -> None:
    """
    Do the image list export using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='export kapture data image list to text file.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='force delete output if already exists.')
    # export ###########################################################################################################
    parser.add_argument('-i', '--input', required=True, help='input path to kapture root folder')
    parser.add_argument('-o', '--output', required=True, help='output path to text file')
    parser.add_argument('-c', '--camera-params', action='store_true', default=False,
                        help='also export camera params in the text file')
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

    export_image_list(args.input, args.output, args.camera_params, args.force)


if __name__ == '__main__':
    export_image_list_command_line()
