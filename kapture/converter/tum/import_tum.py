# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Import a trajectory in the TUM format (as written by GLIM, evo, ORB-SLAM, ...) into a kapture.

TUM format: one pose per line, `timestamp tx ty tz qx qy qz qw`, timestamp in seconds, the pose being the device pose
in the world (world_from_device). Lines starting with # are comments.
"""

import logging
import os.path as path
from decimal import Decimal
from typing import List, Optional, Tuple

import kapture
import kapture.io.csv as csv

logger = logging.getLogger('import_tum')


def read_tum_trajectory(filepath: str) -> List[Tuple[int, kapture.PoseTransform]]:
    """
    Reads a TUM trajectory file.

    :param filepath: path to the TUM file.
    :return: list of (timestamp in microseconds, world_from_device pose), in file order.
    """
    poses = []
    with open(filepath, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            words = line.replace(',', ' ').split()
            if len(words) != 8:
                raise ValueError(f'{filepath}:{line_number}: expected 8 values (t x y z qx qy qz qw), got {len(words)}')
            # exact decimal conversion of the timestamp: a float would lose the microseconds of epoch seconds.
            timestamp_us = int((Decimal(words[0]) * 1_000_000).to_integral_value())
            tx, ty, tz, qx, qy, qz, qw = (float(w) for w in words[1:])
            poses.append((timestamp_us, kapture.PoseTransform(r=[qw, qx, qy, qz], t=[tx, ty, tz])))
    return poses


def tum_to_trajectories(poses: List[Tuple[int, kapture.PoseTransform]],
                        device_id: str,
                        rigs: Optional[kapture.Rigs] = None,
                        rig_id: Optional[str] = None) -> kapture.Trajectories:
    """
    Converts TUM poses (world_from_device) to kapture trajectories (device_from_world), optionally moved to the rig.

    :param poses: (timestamp us, world_from_device) as returned by read_tum_trajectory.
    :param device_id: kapture sensor the TUM poses refer to.
    :param rigs: kapture rigs, needed with rig_id.
    :param rig_id: if given, the trajectory is written for this rig:
                   rig_from_world = rig_from_device o device_from_world.
    :return: the trajectories.
    """
    trajectory_id = device_id
    rig_from_device = None
    if rig_id is not None:
        if rigs is None or rig_id not in rigs or device_id not in rigs[rig_id]:
            raise ValueError(f'{device_id} is not in rig {rig_id}')
        rig_from_device = rigs[rig_id, device_id].inverse()  # rigs hold device_from_rig
        trajectory_id = rig_id
    trajectories = kapture.Trajectories()
    for timestamp_us, world_from_device in poses:
        device_from_world = world_from_device.inverse()
        if rig_from_device is not None:
            pose = kapture.PoseTransform.compose([rig_from_device, device_from_world])
        else:
            pose = device_from_world
        if timestamp_us in trajectories and trajectory_id in trajectories[timestamp_us]:
            logger.warning(f'duplicate pose at {timestamp_us} for {trajectory_id}, last one kept')
        trajectories[timestamp_us, trajectory_id] = pose
    return trajectories


def import_tum_trajectory(tum_filepath: str,
                          kapture_dirpath: str,
                          device_id: str,
                          rig_id: Optional[str] = None,
                          force_overwrite_existing: bool = False) -> kapture.Trajectories:
    """
    Imports a TUM trajectory as the trajectories of an existing kapture.

    :param tum_filepath: input TUM file.
    :param kapture_dirpath: kapture directory, its sensors/trajectories.txt is written.
    :param device_id: kapture sensor the TUM poses refer to (eg. the lidar for GLIM traj_lidar.txt).
    :param rig_id: if given, the poses are converted to this rig with the kapture rigs.
    :param force_overwrite_existing: overwrite an existing trajectories.txt.
    :return: the trajectories written.
    """
    trajectories_filepath = csv.get_csv_fullpath(kapture.Trajectories, kapture_dirpath)
    if path.exists(trajectories_filepath) and not force_overwrite_existing:
        raise FileExistsError(f'{trajectories_filepath} already exists, use force to overwrite')
    sensors = csv.sensors_from_file(csv.get_csv_fullpath(kapture.Sensors, kapture_dirpath))
    if device_id not in sensors:
        raise ValueError(f'unknown sensor {device_id}, kapture sensors: {sorted(sensors)}')
    rigs = None
    if rig_id is not None:
        rigs_filepath = csv.get_csv_fullpath(kapture.Rigs, kapture_dirpath)
        if not path.exists(rigs_filepath):
            raise ValueError(f'rig {rig_id} requested but no rigs in {kapture_dirpath}')
        rigs = csv.rigs_from_file(rigs_filepath)
    poses = read_tum_trajectory(tum_filepath)
    logger.info(f'{len(poses)} poses read from {tum_filepath}')
    if not poses:
        raise ValueError(f'no pose in {tum_filepath}')
    trajectories = tum_to_trajectories(poses, device_id, rigs, rig_id)
    csv.trajectories_to_file(trajectories_filepath, trajectories)
    logger.info(f'{len(trajectories)} poses of {rig_id or device_id} written to {trajectories_filepath}')
    return trajectories
