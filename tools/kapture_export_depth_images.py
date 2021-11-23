#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export a kapture depth maps to PNG files.
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
import kapture.io.features
from kapture.io.records import depth_map_from_file
from typing import Optional, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger('depth2images')


def export_depth_images(
        input_kapture_dir_path: str,
        output_depth_images_dir_path: str,
        depth_range: Optional[Tuple[float, float]] = None,
):
    """

    :param input_kapture_dir_path:
    :param output_depth_images_dir_path:
    :param depth_range: Optional depth range. If not given, automatically adjusted for each frame.
    :return:
    """
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
        if depth_range is None:
            # min max
            depth_range = (depth_map.min(), depth_map.max())

        assert depth_range is not None
        # rescale range
        depth_map = (depth_map - depth_range[0]) / (depth_range[1] - depth_range[0])
        # save to 16bits single channel PNGs
        depth_map = (depth_map * pow(255, 2)).astype(np.uint16)
        depth_image = Image.fromarray(depth_map)
        depth_image.save(output_png_filepath)


def export_depth_images_command_line() -> None:
    """
    export a kapture depth maps to PNG files
    using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='Export depth to images')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-o', '--output',
                        help='output directory. If not given, same as input.')
    parser.add_argument('-r', '--range', nargs='+', type=float,
                        help='min and max value. '
                             'If not given, automatically computed for each frame.')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    if not args.output:
        args.output = args.input
    args.input = path.abspath(args.input)
    args.output = path.abspath(args.output)
    if args.range is not None and len(args.range) != 2:
        parser.error('expect 2 floats as range values.')
    args.range = tuple(args.range)

    print(args.range)
    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v) for k, v in vars(args).items()]))
    export_depth_images(
        input_kapture_dir_path=args.input,
        output_depth_images_dir_path=args.output,
        depth_range=args.range
    )


if __name__ == '__main__':
    export_depth_images_command_line()
