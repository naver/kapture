#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a GLIM (lidar-inertial SLAM) dump into an existing kapture: submap points as the reconstruction,
optionally a TUM trajectory of the dump.
"""

import logging
from typing import Optional

from kapture.converter.glim.import_glim import import_glim
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('import_glim')


def import_glim_dump(input: str,
                     output: str,
                     intensity_gray: bool = False,
                     trajectory: Optional[str] = None,
                     device_id: Optional[str] = None,
                     rig_id: Optional[str] = None,
                     force: bool = False,
                     verbose: Verbosity = DEFAULT_VERBOSITY,
                     quiet: bool = False) -> None:
    """
    Import a GLIM dump directory (glim_rosbag / glim_rosnode output) into an existing kapture: the submap points,
    moved to the world frame, become reconstruction/points3d.txt; with --trajectory, a TUM trajectory of the dump
    (traj_lidar.txt, traj_imu.txt, odom_*.txt) becomes sensors/trajectories.txt.

    :param input: GLIM dump directory.
    :param output: kapture root directory, updated in place.
    :param intensity_gray: color the points in gray from the GLIM intensities (else no color).
    :param trajectory: TUM trajectory file name in the dump to import, eg. traj_lidar.txt.
    :param device_id: kapture sensor the trajectory refers to (eg. the lidar for traj_lidar.txt).
    :param rig_id: write the trajectory for this rig instead of the device.
    :param force: overwrite existing points3d.txt / trajectories.txt.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    import_glim(dump_dirpath=input,
                kapture_dirpath=output,
                intensity_gray=intensity_gray,
                trajectory_filename=trajectory,
                device_id=device_id,
                rig_id=rig_id,
                force_overwrite_existing=force)


def main():
    run_cli(import_glim_dump, aliases={
        '--trajectory': ['-t'],
        '--device_id': ['-d'],
        '--rig_id': ['-r'],
    })


if __name__ == '__main__':
    main()
