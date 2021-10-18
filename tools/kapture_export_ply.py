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
import kapture.io.image as image
import kapture.io.features
from kapture.io.records import depth_map_from_file
from typing import Optional

logger = logging.getLogger('export_ply')


def export_ply(kapture_path: str,  # noqa: C901
               ply_dir_path: str,
               axis_length: float,
               only: Optional[list] = None,
               skip: Optional[list] = None
               ) -> None:
    """
    Export the kapture 3D data in a PLY file.

    :param kapture_path: top directory of the kapture
    :param ply_dir_path: path to the ply file to create
    :param axis_length: length of axis representation (in world unit)
    :param only: list of the only kapture objects to plot (optional)
    :param skip: list of the kapture objects to skip
    """
    try:
        os.makedirs(ply_dir_path, exist_ok=True)

        logger.info('loading data ...')
        with csv.get_all_tar_handlers(kapture_path) as tar_handlers:
            kapture_data = csv.kapture_from_dir(kapture_path, tar_handlers=tar_handlers)

            def _should_do(candidate: str) -> bool:
                pass_only = candidate in only if only else True
                pass_skip = candidate not in skip if skip else False
                return pass_only or pass_skip

            logger.info('exporting  ...')
            if _should_do('rigs') and kapture_data.rigs:
                logger.info(f'creating {len(kapture_data.rigs)} rigs.')
                for rig_id, rig in kapture_data.rigs.items():
                    rig_ply_filepath = path.join(ply_dir_path, f'rig_{rig_id}.ply')
                    logger.info(f'creating rig file : {rig_ply_filepath}.')
                    logger.debug(rig_ply_filepath)
                    ply.rig_to_ply(rig_ply_filepath, rig, axis_length)

            if _should_do('trajectories') and kapture_data.trajectories:
                trajectories_ply_filepath = path.join(ply_dir_path, 'trajectories.ply')
                logger.info(f'creating trajectories file : {trajectories_ply_filepath}')
                ply.trajectories_to_ply(filepath=trajectories_ply_filepath,
                                        trajectories=kapture_data.trajectories,
                                        axis_length=axis_length)

            if _should_do('points3d') and kapture_data.points3d:
                points3d_ply_filepath = path.join(ply_dir_path, 'points3d.ply')
                logger.info(f'creating 3D points file : {points3d_ply_filepath}')
                ply.points3d_to_ply(points3d_ply_filepath, kapture_data.points3d)

            if _should_do('depth') and kapture_data.records_depth:
                logger.info('creating depth maps in 3D.')
                depth_records_filepaths = kapture.io.records.depth_maps_to_filepaths(kapture_data.records_depth,
                                                                                     kapture_path)
                map_depth_to_sensor = {depth_map_name: sensor_id
                                       for _, sensor_id, depth_map_name in kapture.flatten(kapture_data.records_depth)}
                for depth_map_name, depth_map_filepath in tqdm(depth_records_filepaths.items(),
                                                               disable=logger.level >= logging.CRITICAL):
                    depth_png_filepath = path.join(ply_dir_path, f'depth_images/{depth_map_name}.png')
                    logger.debug(f'creating depth map file {depth_png_filepath}')
                    os.makedirs(path.dirname(depth_png_filepath), exist_ok=True)
                    depth_sensor_id = map_depth_to_sensor[depth_map_name]
                    depth_map_sizes = tuple(int(x) for x in kapture_data.sensors[depth_sensor_id].sensor_params[1:3])
                    depth_map = depth_map_from_file(depth_map_filepath, depth_map_sizes)
                    # min max scaling
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    depth_map = (depth_map * 255.).astype(np.uint8)
                    depth_image = Image.fromarray(depth_map)
                    depth_image.save(depth_png_filepath)
            logger.info('done.')
    except Exception as e:
        logging.critical(e)
        raise


def export_ply_command_line() -> None:
    """
    Do the plot to ply file using the parameters given on the command line.
    """
    export_choices = {
        # cmd : help
        'rigs': 'plot the rig geometry, ie. relative pose of sensors into the rig.',
        'rig_stat': 'plot the sensor relative poses for each trajectory timestamp.',
        'trajectories': 'plot the trajectory of every sensors.',
        'points3d': 'plot the 3-D point cloud.',
        'depth': 'plot depth maps as point cloud (one per depth map)'
    }

    parser = argparse.ArgumentParser(description='plot out camera geometry to PLY file.')
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
    parser.add_argument('--only', nargs='+', choices=export_choices.keys(), default=[],
                        help='things to plot : ' + ' // '.join('{}: {}'.format(k, v)
                                                               for k, v in export_choices.items()))
    parser.add_argument('--skip', nargs='+', choices=export_choices.keys(), default=[],
                        help='things to not plot : ' + ' // '.join(export_choices.keys()))
    parser.add_argument('--axis_length', type=float, default=0.1,
                        help='length of axis representation (in world unit).')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    if not args.output:
        args.output = args.input

    # only overwrite skip :
    if args.only:
        args.skip = []
    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v) for k, v in vars(args).items()]))
    export_ply(kapture_path=args.input,
               ply_dir_path=args.output,
               axis_length=args.axis_length,
               only=args.only, skip=args.skip)


if __name__ == '__main__':
    export_ply_command_line()
