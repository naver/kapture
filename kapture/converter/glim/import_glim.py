# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Import a GLIM (https://github.com/koide3/glim) dump directory into a kapture: the submap points as the kapture
reconstruction (reconstruction/points3d.txt), optionally a TUM trajectory of the dump as trajectories.txt.

GLIM dump layout (glim_rosbag / glim_rosnode, SubMap::save):
 - NNNNNN/data.txt: text, with `T_world_origin:` followed by a 4x4 matrix (pose of the submap origin frame).
 - NNNNNN/points_compact.bin: float32 x y z per point, in the submap origin frame.
 - NNNNNN/intensities_compact.bin: float32 per point (optional), normals_compact.bin, covs_compact.bin.
 - traj_lidar.txt, traj_imu.txt (TUM, with loop closure), odom_lidar.txt, odom_imu.txt (odometry only).
"""

import logging
import os
import os.path as path
from typing import List, Optional, Tuple

import numpy as np

import kapture
import kapture.io.csv as csv
from kapture.converter.tum.import_tum import read_tum_trajectory, tum_to_trajectories

logger = logging.getLogger('import_glim')

SUBMAP_DATA_FILENAME = 'data.txt'
SUBMAP_POINTS_FILENAME = 'points_compact.bin'
SUBMAP_INTENSITIES_FILENAME = 'intensities_compact.bin'


def read_submap_matrix(data_filepath: str, name: str = 'T_world_origin') -> np.ndarray:
    """
    Reads a 4x4 matrix of a GLIM submap data.txt.

    :param data_filepath: path to NNNNNN/data.txt.
    :param name: key of the matrix (T_world_origin, T_origin_endpoint_L, ...).
    :return: the 4x4 matrix.
    """
    with open(data_filepath, 'r') as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.strip().rstrip(':') == name:
            rows = [np.array(lines[i + 1 + r].split(), dtype=np.float64) for r in range(4)]
            matrix = np.stack(rows)
            if matrix.shape != (4, 4):
                raise ValueError(f'{data_filepath}: {name} is not a 4x4 matrix')
            return matrix
    raise ValueError(f'{data_filepath}: no {name}')


def read_submap(submap_dirpath: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Reads a GLIM submap.

    :param submap_dirpath: path to the NNNNNN directory.
    :return: T_world_origin (4x4), points (N, 3) in the submap origin frame, intensities (N,) or None.
    """
    world_from_origin = read_submap_matrix(path.join(submap_dirpath, SUBMAP_DATA_FILENAME))
    points = np.fromfile(path.join(submap_dirpath, SUBMAP_POINTS_FILENAME), dtype=np.float32)
    if points.size % 3:
        raise ValueError(f'{submap_dirpath}: {SUBMAP_POINTS_FILENAME} is not a float32 xyz array')
    points = points.reshape(-1, 3)
    intensities = None
    intensities_filepath = path.join(submap_dirpath, SUBMAP_INTENSITIES_FILENAME)
    if path.isfile(intensities_filepath):
        intensities = np.fromfile(intensities_filepath, dtype=np.float32)
        if len(intensities) != len(points):
            logger.warning(f'{submap_dirpath}: {len(intensities)} intensities for {len(points)} points, ignored')
            intensities = None
    return world_from_origin, points, intensities


def list_submaps(dump_dirpath: str) -> List[str]:
    """ :return: the submap directories of a GLIM dump, sorted. """
    submaps = sorted(path.join(dump_dirpath, name) for name in os.listdir(dump_dirpath)
                     if name.isdigit() and path.isfile(path.join(dump_dirpath, name, SUBMAP_DATA_FILENAME)))
    if not submaps:
        raise ValueError(f'{dump_dirpath}: no GLIM submap directory (NNNNNN/{SUBMAP_DATA_FILENAME})')
    return submaps


def intensities_to_gray(intensities: np.ndarray) -> np.ndarray:
    """ :return: (N, 3) gray colors in 0-255, 2nd-98th percentiles of the intensities mapped to 0-255. """
    low, high = np.percentile(intensities, [2, 98])
    if high <= low:
        high = low + 1
    gray = np.clip((intensities - low) / (high - low) * 255., 0, 255)
    return np.repeat(np.round(gray)[:, None], 3, axis=1)


def glim_points3d(dump_dirpath: str, intensity_gray: bool = False) -> kapture.Points3d:
    """
    The submap points of a GLIM dump in the world frame.

    :param dump_dirpath: GLIM dump directory.
    :param intensity_gray: color the points in gray from the intensities (else no color).
    :return: the 3D points.
    """
    all_points, all_intensities = [], []
    for submap_dirpath in list_submaps(dump_dirpath):
        world_from_origin, points, intensities = read_submap(submap_dirpath)
        all_points.append(points.astype(np.float64) @ world_from_origin[:3, :3].T + world_from_origin[:3, 3])
        if intensities is None:
            intensity_gray = False
        all_intensities.append(intensities)
    xyz = np.concatenate(all_points)
    if intensity_gray:
        return kapture.Points3d(np.hstack([xyz, intensities_to_gray(np.concatenate(all_intensities))]))
    return kapture.Points3d(xyz, size=kapture.Points3d.XYZ_ONLY)


def import_glim(dump_dirpath: str,
                kapture_dirpath: str,
                intensity_gray: bool = False,
                trajectory_filename: Optional[str] = None,
                device_id: Optional[str] = None,
                rig_id: Optional[str] = None,
                force_overwrite_existing: bool = False) -> kapture.Points3d:
    """
    Imports a GLIM dump into an existing kapture: submap points as reconstruction/points3d.txt, and optionally a
    TUM trajectory of the dump as sensors/trajectories.txt.

    :param dump_dirpath: GLIM dump directory.
    :param kapture_dirpath: kapture directory, updated in place.
    :param intensity_gray: color the points in gray from the GLIM intensities.
    :param trajectory_filename: TUM trajectory of the dump to import (eg. traj_lidar.txt), none by default.
    :param device_id: kapture sensor the trajectory refers to (lidar for traj_lidar.txt, IMU for traj_imu.txt).
    :param rig_id: write the trajectory for this rig (with the kapture rigs) instead of the device.
    :param force_overwrite_existing: overwrite existing points3d.txt / trajectories.txt.
    :return: the 3D points written.
    """
    points3d_filepath = csv.get_csv_fullpath(kapture.Points3d, kapture_dirpath)
    if path.exists(points3d_filepath) and not force_overwrite_existing:
        raise FileExistsError(f'{points3d_filepath} already exists, use force to overwrite')
    trajectories = None
    if trajectory_filename is not None:
        if device_id is None:
            raise ValueError('device_id is needed to import the trajectory')
        trajectories_filepath = csv.get_csv_fullpath(kapture.Trajectories, kapture_dirpath)
        if path.exists(trajectories_filepath) and not force_overwrite_existing:
            raise FileExistsError(f'{trajectories_filepath} already exists, use force to overwrite')
        sensors = csv.sensors_from_file(csv.get_csv_fullpath(kapture.Sensors, kapture_dirpath))
        if device_id not in sensors:
            raise ValueError(f'unknown sensor {device_id}, kapture sensors: {sorted(sensors)}')
        rigs = None
        if rig_id is not None:
            rigs = csv.rigs_from_file(csv.get_csv_fullpath(kapture.Rigs, kapture_dirpath))
        poses = read_tum_trajectory(path.join(dump_dirpath, trajectory_filename))
        trajectories = tum_to_trajectories(poses, device_id, rigs, rig_id)

    points3d = glim_points3d(dump_dirpath, intensity_gray)
    csv.points3d_to_file(points3d_filepath, points3d)
    logger.info(f'{len(points3d)} GLIM submap points written to {points3d_filepath}')
    if trajectories is not None:
        csv.trajectories_to_file(trajectories_filepath, trajectories)
        logger.info(f'{len(trajectories)} poses of {rig_id or device_id} written to {trajectories_filepath}')
    return points3d
