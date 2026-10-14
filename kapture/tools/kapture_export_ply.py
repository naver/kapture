#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to plot a kapture to a PLY file.
"""

import logging
import os
import os.path as path
import sys

from tqdm import tqdm
import numpy as np
import kapture
import kapture.io.csv as csv
import kapture.io.ply as ply
import kapture.io.features
from typing import List, Literal
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('ply')

try:
    import open3d as o3d
except (ImportError, OSError) as e:
    # postpone warning to the actual use.
    logger.debug(f'cant import {e}')

CHOICE_RIGS = 'rigs'
CHOICE_TRAJECTORIES = 'trajectories'
CHOICE_POINTS3D = 'points3d'
CHOICE_LIDAR = 'lidar'

export_choices = {
    CHOICE_RIGS: 'plot the rig geometry, ie. relative pose of sensors into the rig.',
    CHOICE_TRAJECTORIES: 'plot the trajectory of every sensors.',
    CHOICE_POINTS3D: 'plot the 3-D point cloud.',
    CHOICE_LIDAR: 'plot depth maps as point cloud (one per depth map)'
}
ExportChoice = Literal['rigs', 'trajectories', 'points3d', 'lidar']


def guess_what_to_do(
        choices: List[str],
        only: List[str] = [],
        skip: List[str] = []
) -> List[str]:
    if only:
        choices = [c for c in choices if c in only]
    if skip:
        choices = [c for c in choices if c not in skip]
    return choices


def export_ply(input: str,  # noqa: C901
               output: str = '',
               axis_length: float = 0.1,
               only: List[ExportChoice] = [],
               skip: List[ExportChoice] = [],
               verbose: Verbosity = DEFAULT_VERBOSITY,
               quiet: bool = False
               ) -> None:
    """
    Export 3D points data (rigs: plot the rig geometry, ie. relative pose of sensors into the rig;
    trajectories: plot the trajectory of every sensors; points3d: plot the 3-D point cloud;
    lidar: plot depth maps as point cloud, one per depth map) to PLY files.

    :param input: input path to kapture data root directory
    :param output: output directory (PLY file format). If not given, same as input.
    :param axis_length: length of axis representation (in world unit)
    :param only: things to plot
    :param skip: things to not plot
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    input = path.abspath(input)
    output = path.abspath(output) if output else input
    try:
        os.makedirs(output, exist_ok=True)

        what_to_do = guess_what_to_do(choices=export_choices.keys(), only=only, skip=skip)

        logger.info('loading data ...')
        data_skip_list = []
        if CHOICE_TRAJECTORIES not in what_to_do:
            data_skip_list.append(kapture.Trajectories)
        if CHOICE_POINTS3D not in what_to_do:
            data_skip_list.append(kapture.Points3d)
        if CHOICE_LIDAR not in what_to_do:
            data_skip_list.append(kapture.RecordsLidar)
        with csv.get_all_tar_handlers(input) as tar_handlers:
            kapture_data = csv.kapture_from_dir(input, skip_list=data_skip_list, tar_handlers=tar_handlers)

        logger.info('exporting  ...')
        if CHOICE_RIGS in what_to_do and kapture_data.rigs:
            logger.info(f'creating {len(kapture_data.rigs)} rigs.')
            for rig_id, rig in kapture_data.rigs.items():
                rig_ply_filepath = path.join(output, f'rig_{rig_id}.ply')
                logger.info(f'creating rig file : {rig_ply_filepath}.')
                logger.debug(rig_ply_filepath)
                ply.rig_to_ply(rig_ply_filepath, rig, axis_length)

        if CHOICE_TRAJECTORIES in what_to_do and kapture_data.trajectories:
            trajectories_ply_filepath = path.join(output, 'trajectories.ply')
            logger.info(f'creating trajectories file : {trajectories_ply_filepath}')
            ply.trajectories_to_ply(filepath=trajectories_ply_filepath,
                                    trajectories=kapture_data.trajectories,
                                    axis_length=axis_length)

        if CHOICE_POINTS3D in what_to_do and kapture_data.points3d:
            points3d_ply_filepath = path.join(output, 'points3d.ply')
            logger.info(f'creating 3D points file : {points3d_ply_filepath}')
            ply.points3d_to_ply(points3d_ply_filepath, kapture_data.points3d)

        if CHOICE_LIDAR in what_to_do and kapture_data.records_lidar:
            lidar_output = path.join(output, 'records_data')
            logger.info(f'creating lidar points files : {lidar_output}')
            if kapture_data.rigs and kapture_data.trajectories:
                # compute trajectories for lidars only
                lidars = [sensor_id for sensor_id, sensor in kapture_data.sensors.items()
                          if sensor.sensor_type == kapture.SensorType.lidar.name]
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
                lidar_src_file_path = kapture.io.records.get_record_fullpath(input, lidar_src_file_name)
                lidar_dst_file_path = path.join(lidar_output, lidar_id, f'{timestamp}.ply')
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


def main():
    run_cli(export_ply)


if __name__ == '__main__':
    main()
