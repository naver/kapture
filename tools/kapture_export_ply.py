#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to plot a kapture to a PLY file.
"""

import argparse
import logging
import os
import os.path as path
import sys

from tqdm import tqdm
import numpy as np
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.csv as csv
import kapture.io.ply as ply
import kapture.io.features
from typing import Optional, List

logger = logging.getLogger('ply')

try:
    import open3d as o3d
except ImportError as e:
    # postpone warning to the actual use.
    logger.debug(f'cant import {e}')


export_choices = {
    'rigs': 'plot the rig geometry, ie. relative pose of sensors into the rig.',
    'trajectories': 'plot the trajectory of every sensors.',
    'points3d': 'plot the 3-D point cloud.',
    'lidar': 'plot depth maps as point cloud (one per depth map)'
}


def guess_what_to_do(
        choices: List[str],
        only: Optional[list] = None,
        skip: Optional[list] = None
) -> List[str]:
    if only:
        choices = [c for c in choices if c in only]
    if skip:
        choices = [c for c in choices if c not in skip]
    return choices


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

        what_to_do = guess_what_to_do(choices=export_choices.keys(), only=only, skip=skip)

        logger.info('loading data ...')
        with csv.get_all_tar_handlers(kapture_path) as tar_handlers:
            kapture_data = csv.kapture_from_dir(kapture_path, tar_handlers=tar_handlers)

        logger.info('exporting  ...')
        if 'rigs' in what_to_do and kapture_data.rigs:
            logger.info(f'creating {len(kapture_data.rigs)} rigs.')
            for rig_id, rig in kapture_data.rigs.items():
                rig_ply_filepath = path.join(ply_dir_path, f'rig_{rig_id}.ply')
                logger.info(f'creating rig file : {rig_ply_filepath}.')
                logger.debug(rig_ply_filepath)
                ply.rig_to_ply(rig_ply_filepath, rig, axis_length)

        if 'trajectories' in what_to_do and kapture_data.trajectories:
            trajectories_ply_filepath = path.join(ply_dir_path, 'trajectories.ply')
            logger.info(f'creating trajectories file : {trajectories_ply_filepath}')
            ply.trajectories_to_ply(filepath=trajectories_ply_filepath,
                                    trajectories=kapture_data.trajectories,
                                    axis_length=axis_length)

        if 'points3d' in what_to_do and kapture_data.points3d:
            points3d_ply_filepath = path.join(ply_dir_path, 'points3d.ply')
            logger.info(f'creating 3D points file : {points3d_ply_filepath}')
            ply.points3d_to_ply(points3d_ply_filepath, kapture_data.points3d)

        if 'lidar' in what_to_do and kapture_data.records_lidar:
            lidar_ply_dir_path = path.join(ply_dir_path, 'records_data')
            logger.info(f'creating lidar points files : {lidar_ply_dir_path}')
            if kapture_data.rigs and kapture_data.trajectories:
                # compute trajectories for lidars only
                lidars = [sensor_id for sensor_id, sensor in kapture_data.sensors.items()
                          if sensor.sensor_type == "lidar"]
                rigs_lidars_only = kapture.Rigs()
                for rig_id, sensor_id, pose in kapture.flatten(kapture_data.rigs):
                    if sensor_id in lidars:
                        rigs_lidars_only[rig_id, sensor_id] = pose

                trajectories_lidars_only = kapture.rigs_remove(kapture_data.trajectories, rigs_lidars_only)
            else:
                trajectories_lidars_only = kapture_data.trajectories

            lidars = list(kapture.flatten(kapture_data.records_lidar, is_sorted=True))
            # lidar export requires open3d: warn user
            if lidars and 'open3d' not in sys.modules:
                logger.critical('exporting lidar point cloud requires the python package open3d. '
                                'If you want lidar pcd to be converted, install this module.'
                                'If you want to silence this message, uses --skip lidar.')
                lidars = []  # nice skip

            hide_progress = logger.getEffectiveLevel() > logging.INFO
            for timestamp, lidar_id, lidar_src_file_name in tqdm(lidars, disable=hide_progress):
                # source lidar file may be ply or pcd
                lidar_src_file_path = kapture.io.records.get_record_fullpath(kapture_path, lidar_src_file_name)
                lidar_dst_file_path = path.join(lidar_ply_dir_path, lidar_id, f'{timestamp}.ply')
                os.makedirs(path.dirname(lidar_dst_file_path), exist_ok=True)

                points3d_o3d = o3d.io.read_point_cloud(lidar_src_file_path)
                # switch to np array to apply world transform
                points3d_np = np.asarray(points3d_o3d.points)
                if timestamp not in trajectories_lidars_only or lidar_id not in trajectories_lidars_only[timestamp]:
                    logger.debug(f'pose not found for lidar "{lidar_id}" records at time {timestamp}.')
                    continue
                pose_lidar_from_world = trajectories_lidars_only[timestamp, lidar_id]
                pose_world_from_lidar = pose_lidar_from_world.inverse()
                points3d_np = pose_world_from_lidar.transform_points(points3d_np)
                # switch back to open3d to save
                points3d_o3d = o3d.geometry.PointCloud()
                points3d_o3d.points = o3d.utility.Vector3dVector(points3d_np)
                o3d.io.write_point_cloud(lidar_dst_file_path, points3d_o3d)
                del points3d_o3d

            logger.info('done.')
    except Exception as e:
        logging.critical(e)
        raise


def export_ply_command_line() -> None:
    """
    Do the plot to ply file using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description=f'Export 3D points data ({", ".join(export_choices.keys())})'
                                                 f' to ply files: ' +
                                                 ' // '.join(f'{k}: {v}' for k, v in export_choices.items()))
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
                        help='things to plot : ' + ', '.join('{}: {}'.format(k, v)
                                                             for k, v in export_choices.items()))
    parser.add_argument('--skip', nargs='+', choices=export_choices.keys(), default=[],
                        help='things to not plot : ' + ', '.join(export_choices.keys()))
    parser.add_argument('--axis_length', type=float, default=0.1,
                        help='length of axis representation (in world unit).')
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
    export_ply(kapture_path=args.input,
               ply_dir_path=args.output,
               axis_length=args.axis_length,
               only=args.only, skip=args.skip)


if __name__ == '__main__':
    export_ply_command_line()
