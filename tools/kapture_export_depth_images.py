#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to plot a kapture to a PLY file.
"""

import argparse
import logging
import os
import os.path as path
from tqdm import tqdm
import numpy as np
from PIL import Image
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.csv as csv
import kapture.io.ply as ply
import kapture.io.features
from kapture.io.records import depth_map_from_file
from typing import Optional, List

logger = logging.getLogger('depth2images')


def export_depth_images(
        input_kapture_dir_path: str,
        output_depth_images_dir_path: str,
        auto_depth_range: bool = True
):
    logger.info('loading data ...')
    with csv.get_all_tar_handlers(input_kapture_dir_path) as tar_handlers:
        kapture_data = csv.kapture_from_dir(input_kapture_dir_path, tar_handlers=tar_handlers)

    logger.info('exporting  ...')

    depth_records_file_paths = kapture.io.records.depth_maps_to_filepaths(
        kapture_data.records_depth, input_kapture_dir_path)

    sensor_ids = {
        depth_map_name: sensor_id
        for _, sensor_id, depth_map_name in kapture.flatten(kapture_data.records_depth)}

    for depth_map_name, depth_map_filepath in tqdm(depth_records_file_paths.items(),
                                                   disable=logger.level >= logging.CRITICAL):
        output_png_filepath = path.join(output_depth_images_dir_path, f'{depth_map_name}.png')
        logger.debug(f'creating file {output_png_filepath}')
        os.makedirs(path.dirname(output_png_filepath), exist_ok=True)
        sensor_id = sensor_ids[depth_map_name]
        map_sizes = tuple(int(x) for x in kapture_data.sensors[sensor_id].sensor_params[1:3])
        depth_map = depth_map_from_file(depth_map_filepath, map_sizes)
        # min max scaling
        if auto_depth_range:
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

        depth_map = (depth_map * 255.).astype(np.uint8)
        depth_image = Image.fromarray(depth_map)
        depth_image.save(output_png_filepath)


def export_depth_images_command_line() -> None:
    """
    Do the plot to ply file using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description=f'Export depth to images')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-o', '--output', required=False,
                        help='output directory (PLY file format).')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    if not args.output:
        args.output = args.input

    args.input = path.abspath(args.input)
    args.output = path.abspath(args.output)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v) for k, v in vars(args).items()]))
    export_depth_images(
        input_kapture_dir_path=args.input,
        output_depth_images_dir_path=args.output)


if __name__ == '__main__':
    export_depth_images_command_line()
