#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export a kapture depth maps to PNG files.
"""

import logging
import os
import os.path as path
from tqdm import tqdm
import numpy as np
from PIL import Image
import kapture
import kapture.io.csv as csv
import kapture.io.features
from kapture.io.records import depth_map_from_file
from typing import List, Optional
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('depth2images')


def export_depth_images(
        input: str,
        output: Optional[str] = None,
        depth_range: Optional[List[float]] = None,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False,
):
    """
    Export kapture depth maps to 16 bits PNG images.

    :param input: input path to kapture data root directory
    :param output: output directory. If not given, same as input.
    :param depth_range: min and max depth values. If not given, automatically computed for each frame.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    input = path.abspath(input)
    output = path.abspath(output) if output else input
    if depth_range is not None:
        if len(depth_range) != 2:
            raise ValueError('expect 2 floats as depth range values.')
        depth_range = tuple(depth_range)
    logger.info('loading data ...')
    with csv.get_all_tar_handlers(input) as tar_handlers:
        kapture_data = csv.kapture_from_dir(input, tar_handlers=tar_handlers)

    logger.info('exporting  ...')

    depth_records_file_paths = kapture.io.records.depth_maps_to_filepaths(
        kapture_data.records_depth, input)

    sensor_ids = {
        depth_map_name: sensor_id
        for _, sensor_id, depth_map_name in kapture.flatten(kapture_data.records_depth)}

    for depth_map_name, depth_map_filepath in tqdm(depth_records_file_paths.items(),
                                                   disable=logger.level >= logging.CRITICAL):
        output_png_filepath = path.join(output, f'{depth_map_name}.png')
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


def main():
    run_cli(export_depth_images, aliases={'--depth_range': ['-r', '--range']})


if __name__ == '__main__':
    main()
