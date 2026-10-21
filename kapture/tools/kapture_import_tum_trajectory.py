#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a TUM format trajectory (eg. from GLIM, evo, ORB-SLAM) as the trajectories of an existing kapture.
"""

import logging
from typing import Optional

from kapture.converter.tum.import_tum import import_tum_trajectory
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('import_tum')


def import_tum(input: str,
               output: str,
               device_id: str,
               rig_id: Optional[str] = None,
               force: bool = False,
               verbose: Verbosity = DEFAULT_VERBOSITY,
               quiet: bool = False) -> None:
    """
    Import a TUM trajectory file (lines: timestamp_s tx ty tz qx qy qz qw, device pose in the world) as the
    sensors/trajectories.txt of an existing kapture. Poses are converted to kapture (world to device), and to the rig
    if rig_id is given, with the kapture rigs.

    :param input: input TUM trajectory file.
    :param output: kapture root directory, its trajectories.txt is written.
    :param device_id: kapture sensor the TUM poses refer to (eg. the lidar for GLIM traj_lidar.txt).
    :param rig_id: write the trajectory for this rig instead of the device.
    :param force: overwrite an existing trajectories.txt.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    import_tum_trajectory(tum_filepath=input,
                          kapture_dirpath=output,
                          device_id=device_id,
                          rig_id=rig_id,
                          force_overwrite_existing=force)


def main():
    run_cli(import_tum, aliases={
        '--device_id': ['-d'],
        '--rig_id': ['-r'],
    })


if __name__ == '__main__':
    main()
