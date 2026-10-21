#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Script to export kapture lidar, IMU and camera records to a ROS2 bag (for lidar-inertial SLAMs like GLIM, FAST-LIO2).
"""

import logging
from typing import List, Literal, Optional

from kapture.converter.ros_tools.export_ros2bag import kapture_to_ros2bag
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('export_ros2bag')


def export_ros2bag(input: str,
                   output: str,
                   lidar_ids: Optional[List[str]] = None,
                   imu_ids: Optional[List[str]] = None,
                   camera_ids: List[str] = [],
                   time_field: str = 'timestamps_0',
                   intensity_field: str = 'intensities_0',
                   storage: Literal['sqlite3', 'mcap'] = 'sqlite3',
                   compression: Literal['none', 'zstd'] = 'none',
                   force: bool = False,
                   verbose: Verbosity = DEFAULT_VERBOSITY,
                   quiet: bool = False) -> None:
    """
    Export kapture lidar, IMU and camera records to a ROS2 bag, written in pure python (needs the rosbags package).
    Topics: /<lidar_id>/points (PointCloud2 with x y z intensity t), /<imu_id>/imu (Imu from the <imu_id>_accel and
    <imu_id>_gyro sensors), /tf_static (rigs), /<camera_id>/image/compressed (+ camera_info for pinhole models).
    The lidar-IMU extrinsics are logged in TUM order [x y z qx qy qz qw] for the SLAM configuration.

    :param input: input path to kapture root directory.
    :param output: output ROS2 bag directory.
    :param lidar_ids: lidars to export (default: all lidars with records).
    :param imu_ids: IMUs to export, <imu_id> for the sensors <imu_id>_accel and <imu_id>_gyro (default: all).
    :param camera_ids: cameras to export, as jpeg/png CompressedImage (default: none).
    :param time_field: per-point time property (ns) of the lidar point cloud files, empty for none.
    :param intensity_field: intensity property of the lidar point cloud files, empty for none.
    :param storage: rosbag2 storage plugin (sqlite3 is readable by every ROS2 distribution, mcap needs the
                    rosbag2_storage_mcap plugin on the ROS2 side).
    :param compression: per message compression.
    :param force: silently delete the output bag if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    kapture_to_ros2bag(kapture_dirpath=input,
                       bag_dirpath=output,
                       lidar_ids=lidar_ids,
                       imu_ids=imu_ids,
                       camera_ids=camera_ids,
                       time_field=time_field or None,
                       intensity_field=intensity_field or None,
                       storage=storage,
                       compression=compression,
                       force_overwrite_existing=force)


def main():
    run_cli(export_ros2bag, aliases={
        '--lidar_ids': ['-l'],
        '--imu_ids': ['-u'],
        '--camera_ids': ['-c'],
    })


if __name__ == '__main__':
    main()
