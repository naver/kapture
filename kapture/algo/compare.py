# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Distances computation and comparison of kapture objects
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Any, Union

import kapture
from kapture import flatten
from kapture.core.Sensors import Camera
from kapture.utils.logging import getLogger

from .pose_operations import pose_transform_distance


def float_iszero(distance: float, threshold: float = 1e-05) -> bool:
    """
    Computes if a distance is close to zero modulo an epsilon.

    :param distance: distance to evaluate
    :param threshold: the epsilon value
    :return: true or false if the condition is met
    """
    return math.isclose(distance, 0.0, rel_tol=threshold, abs_tol=threshold)


def is_distance_within_threshold(pose_distance: Tuple[float, float],
                                 pose_thresholds: Tuple[float, float] = (1e-05, 1e-05)
                                 ) -> bool:
    """
    compare a pose distance tuple to (0,0) with some thresholds

    :param pose_distance: (translation_distance, rotation_distance)
    :type pose_distance: Tuple[float, float]
    :param pose_thresholds: (translation_threshold, rotation_threshold)
    :type pose_thresholds: Tuple[float, float], optional
    :return: True if both rotation and translation distance are within threshold
    :rtype: bool
    """
    translation_distance, rotation_distance = pose_distance
    translation_threshold, rotation_threshold = pose_thresholds
    return float_iszero(translation_distance, translation_threshold) \
        and float_iszero(rotation_distance, rotation_threshold)


def equal_poses(pose_a: kapture.PoseTransform, pose_b: kapture.PoseTransform) -> bool:
    """
    Compare the two pose to check if they are equal.

    :param pose_a: first pose
    :param pose_b: second pose
    :return: True if they are equal, False otherwise
    """
    pose_a_nones = (pose_a.r is None, pose_a.t is None)
    pose_b_nones = (pose_b.r is None, pose_b.t is None)
    if pose_a_nones != pose_b_nones:
        return False

    pose_distance = pose_transform_distance(pose_a, pose_b)
    if pose_a_nones == (True, True):  # a and b have None rotation, None translation
        return True
    elif pose_a_nones == (True, False):  # a and b have None rotation, valid translation
        return float_iszero(pose_distance[0])
    elif pose_a_nones == (False, True):  # a and b have valid rotation, None translation
        return float_iszero(pose_distance[1])
    else:
        return is_distance_within_threshold(pose_distance)


def equal_camera_params(camera_params_a: List[float], camera_params_b: List[float]) -> bool:
    """
    Checks if the camera parameters are equals.

    :param camera_params_a: first camera parameters
    :param camera_params_b: second camera parameters
    :return: True if they are equal, False otherwise
    """
    return np.isclose([float(v) for v in camera_params_a], [float(v) for v in camera_params_b]).all()


def equal_sensors(sensors_a: Optional[kapture.Sensors], sensors_b: Optional[kapture.Sensors]) -> bool:
    """
    Compare two instances of kapture.Sensors.
    model_params for cameras are considered equal if np.isclose says so.

    :param sensors_a: first sensor definition
    :param sensors_b: second sensor definition
    :return: True if they are identical, False otherwise.
    """
    if sensors_a is None and sensors_b is None:
        return True
    elif sensors_a is None and sensors_b is not None:
        return False
    elif sensors_a is not None and sensors_b is None:
        return False

    flattened_a = list(flatten(sensors_a, is_sorted=True))
    flattened_b = list(flatten(sensors_b, is_sorted=True))
    if len(flattened_a) != len(flattened_b):
        getLogger().debug('equal_sensors: a and b do not have the same number of elements')
        return False
    for (sensor_id_a, sensor_a), (sensor_id_b, sensor_b) in zip(flattened_a, flattened_b):
        # handling special case: name_a='' and name_b=None
        equal_id = sensor_id_a == sensor_id_b
        equal_name = (not sensor_a.name and not sensor_b.name) or (sensor_a.name == sensor_b.name)
        equal_type = sensor_a.sensor_type == sensor_b.sensor_type

        if not equal_id or not equal_name or not equal_type:
            getLogger().debug(
                f'equal_sensors: ({sensor_id_a}, {sensor_a}) != ({sensor_id_b}, {sensor_b})')
            return False

        equal_params = False
        if sensor_a.sensor_type == 'camera':
            assert isinstance(sensor_a, Camera)
            assert isinstance(sensor_b, Camera)
            if sensor_a.camera_type == sensor_b.camera_type:
                equal_params = equal_camera_params(sensor_a.camera_params, sensor_b.camera_params)
        else:
            equal_params = sensor_a.sensor_params == sensor_b.sensor_params

        if not equal_params:
            getLogger().debug(
                f'equal_sensors: ({sensor_id_a}, {sensor_a}) != ({sensor_id_b}, {sensor_b})')
            return False
    return True


def equal_rigs(rigs_a: Optional[kapture.Rigs], rigs_b: Optional[kapture.Rigs]) -> bool:
    """
    Compare two instances of kapture.Rigs.
    Poses are compared with is_distance_within_threshold(pose_transform_distance())

    :param rigs_a: first set of rigs
    :param rigs_b: second set of rigs
    :return: True if they are identical, False otherwise.
    """
    if rigs_a is None and rigs_b is None:
        return True
    elif rigs_a is None and rigs_b is not None:
        return False
    elif rigs_a is not None and rigs_b is None:
        return False

    flattened_a = list(flatten(rigs_a, is_sorted=True))
    flattened_b = list(flatten(rigs_b, is_sorted=True))
    if len(flattened_a) != len(flattened_b):
        getLogger().debug('equal_rigs: a and b do not have the same number of elements')
        return False
    for (rig_id_a, sensor_id_a, pose_a), (rig_id_b, sensor_id_b, pose_b) in zip(flattened_a, flattened_b):
        if rig_id_a != rig_id_b or sensor_id_a != sensor_id_b:
            getLogger().debug(
                f'equal_rigs: ({rig_id_a}, {sensor_id_a}, {pose_a.r_raw}, {pose_a.t_raw}) !='
                f' ({rig_id_b}, {sensor_id_b}, {pose_b.r_raw}, {pose_b.t_raw})')
            return False
        if not equal_poses(pose_a, pose_b):
            getLogger().debug(
                f'equal_rigs: ({rig_id_a}, {sensor_id_a}, {pose_a.r_raw}, {pose_a.t_raw}) '
                f'is not close to '
                f'({rig_id_b}, {sensor_id_b}, {pose_b.r_raw}, {pose_b.t_raw})')
            return False
    return True


def equal_trajectories(trajectories_a: Optional[kapture.Trajectories],
                       trajectories_b: Optional[kapture.Trajectories]) -> bool:
    """
    Compare two instances of kapture.Trajectories.
    Poses are compared with is_distance_within_threshold(pose_transform_distance())

    :param trajectories_a: first trajectory
    :param trajectories_b: second trajectory
    :return: True if they are identical, False otherwise.
    """
    if trajectories_a is None and trajectories_b is None:
        return True
    elif trajectories_a is None and trajectories_b is not None:
        return False
    elif trajectories_a is not None and trajectories_b is None:
        return False

    flattened_a = list(flatten(trajectories_a, is_sorted=True))
    flattened_b = list(flatten(trajectories_b, is_sorted=True))
    if len(flattened_a) != len(flattened_b):
        getLogger().debug('equal_trajectories: a and b do not have the same number of elements')
        return False
    for (timestamp_a, sensor_id_a, pose_a), (timestamp_b, sensor_id_b, pose_b) in zip(flattened_a, flattened_b):
        if timestamp_a != timestamp_b or sensor_id_a != sensor_id_b:
            getLogger().debug(
                f'equal_trajectories: ({timestamp_a}, {sensor_id_a}, {pose_a.r_raw}, {pose_a.t_raw}) !='
                f' ({timestamp_b}, {sensor_id_b}, {pose_b.r_raw}, {pose_b.t_raw})')
            return False
        if not equal_poses(pose_a, pose_b):
            getLogger().debug(
                f'equal_trajectories: ({timestamp_a}, {sensor_id_a}, {pose_a.r_raw}, {pose_a.t_raw}) '
                f'is not close to '
                f'({timestamp_b}, {sensor_id_b}, {pose_b.r_raw}, {pose_b.t_raw})')
            return False
    return True


def log_difference(a: List[Tuple[Any, ...]], b: List[Tuple[Any, ...]], func_name: str, trim_count: int = 5) -> None:
    """
    Records in the logger the difference between two values.

    :param a: first value
    :param b: second value
    :param func_name: comparison function to print
    :param trim_count: maximum number of values to record
    """
    if len(a) != len(b):
        getLogger().debug(f'{func_name}: a and b do not have the same number of elements')
    else:
        diffs = [(va, vb) for va, vb in zip(a, b) if va != vb]
        diffs = diffs[:trim_count]
        diffs = ['({}) != ({})'.format(', '.join([str(f) for f in va]),
                                       ', '.join([str(f) for f in vb]))
                 for va, vb in diffs]
        getLogger().debug('{}:\n{}'.format(func_name, '\n'.join(diffs)))


def equal_nested_dict_or_set(data_a, data_b, name_to_log) -> bool:
    """
    Compare two instances of dictionary or set

    :return: True if they are identical, False otherwise.
    """
    if data_a is None and data_b is None:
        return True
    elif data_a is None and data_b is not None:
        return False
    elif data_a is not None and data_b is None:
        return False

    flattened_a = list(flatten(data_a, is_sorted=True))
    flattened_b = list(flatten(data_b, is_sorted=True))
    are_equal = (flattened_a == flattened_b)
    if not are_equal:
        log_difference(flattened_a, flattened_b, name_to_log)
    return are_equal


def equal_image_features(data_a: Optional[Union[kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures]],
                         data_b: Optional[Union[kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures]]
                         ) -> bool:
    """
    Compare two instances of kapture features (keypoints, descriptors or global features).

    :param data_a: first set of features
    :param data_b: second set of features
    :return: True if they are identical, False otherwise.
    """
    if data_a is None and data_b is None:
        return True
    elif data_a is None and data_b is not None:
        return False
    elif data_a is not None and data_b is None:
        return False

    # should not happen because of previous lines, use assert to help your ide figure out the type of data
    assert data_a is not None
    assert data_b is not None

    if data_a.type_name != data_b.type_name or data_a.dtype != data_b.dtype or data_a.dsize != data_b.dsize:
        return False
    flattened_a = list(flatten(data_a, is_sorted=True))
    flattened_b = list(flatten(data_b, is_sorted=True))
    are_equal = (flattened_a == flattened_b)
    if not are_equal:
        log_difference(flattened_a, flattened_b, 'equal_image_features')
    return are_equal


def equal_records_camera(records_a: Optional[kapture.RecordsCamera],
                         records_b: Optional[kapture.RecordsCamera]) -> bool:
    """
    Compare two instances of kapture.RecordsCamera.

    :param records_a: first set of records
    :param records_b: second set of records
    :return: True if they are identical, False otherwise.
    """
    return equal_nested_dict_or_set(records_a, records_b, 'equal_records_camera')


def equal_records_lidar(records_a: Optional[kapture.RecordsLidar],
                        records_b: Optional[kapture.RecordsLidar]) -> bool:
    """
    Compare two instances of kapture.RecordsLidar.

    :param records_a: first set of records
    :param records_b: second set of records
    :return: True if they are identical, False otherwise.
    """
    return equal_nested_dict_or_set(records_a, records_b, 'equal_records_lidar')


def equal_records_wifi(records_a: Optional[kapture.RecordsWifi],
                       records_b: Optional[kapture.RecordsWifi]) -> bool:
    """
    Compare two instances of kapture.RecordsWifi.

    :param records_a: first set of records
    :param records_b: second set of records
    :return: True if they are identical, False otherwise.
    """
    return equal_nested_dict_or_set(records_a, records_b, 'equal_records_wifi')


def equal_records_gnss(records_a: Optional[kapture.RecordsGnss],
                       records_b: Optional[kapture.RecordsGnss]) -> bool:
    """
    Compare two instances of kapture.RecordsGnss.

    :param records_a: first set of records
    :param records_b: second set of records
    :return: True if they are identical, False otherwise.
    """
    return equal_nested_dict_or_set(records_a, records_b, 'equal_records_gnss')


def equal_matches(matches_a: Optional[kapture.Matches],
                  matches_b: Optional[kapture.Matches]) -> bool:
    """
    Compare two instances of kapture.Matches.

    :param matches_a: first set of matches
    :param matches_b: second set of matches
    :return: True if they are identical, False otherwise.
    """
    return equal_nested_dict_or_set(matches_a, matches_b, 'equal_matches')


def equal_observations(data_a: Optional[kapture.Observations],
                       data_b: Optional[kapture.Observations]) -> bool:
    """
    Compare two instances of kapture.Observations.

    :param data_a: first set of observations
    :param data_b: second set of observations
    :return: True if they are identical, False otherwise.
    """
    return equal_nested_dict_or_set(data_a, data_b, 'equal_observations')


def equal_points3d(points3d_a: Optional[kapture.Points3d],
                   points3d_b: Optional[kapture.Points3d]) -> bool:
    """
    Compare two instances of kapture.Points3d.

    :param points3d_a: first set of points3d
    :param points3d_b: second set of points3d
    :return: True if they are identical, False otherwise.
    """
    if points3d_a is None and points3d_b is None:
        return True
    elif points3d_a is None and points3d_b is not None:
        return False
    elif points3d_a is not None and points3d_b is None:
        return False

    # ide guidance
    assert points3d_a is not None
    assert points3d_b is not None

    if len(points3d_a) != len(points3d_b):
        getLogger().debug('equal_points3d: a and b do not have the same number of elements')

    # internally converted to array of bool which cannot be a point3d
    bool_array = np.isclose(points3d_a.as_array(), points3d_b.as_array())
    are_equal = bool_array.all()
    if not are_equal:
        diffs = [n for n, b in enumerate(bool_array) if not b]
        diffs = diffs[:15]
        diffs = ['element {} : {} != {}'.format(n, points3d_a[n], points3d_b[n]) for n in diffs]
        getLogger().debug('equal_points3d:\n{}'.format('\n'.join(diffs)))
    return are_equal


def equal_kapture(data_a: kapture.Kapture, data_b: kapture.Kapture) -> bool:
    """
    Compare two instances of Kapture.
     Poses are compared with is_distance_within_threshold(pose_transform_distance())

    :param data_a: first kapture
    :param data_b: second kapture
    :return: True if they are identical, False otherwise.
    """
    # compare sensors
    if not equal_sensors(data_a.sensors, data_b.sensors):
        return False

    # compare rigs
    if not equal_rigs(data_a.rigs, data_b.rigs):
        return False

    # compare trajectories
    if not equal_trajectories(data_a.trajectories, data_b.trajectories):
        return False

    # compare records
    if not equal_records_camera(data_a.records_camera, data_b.records_camera):
        return False
    if not equal_records_lidar(data_a.records_lidar, data_b.records_lidar):
        return False
    if not equal_records_wifi(data_a.records_wifi, data_b.records_wifi):
        return False
    if not equal_records_gnss(data_a.records_gnss, data_b.records_gnss):
        return False

    # compare image features (keypoints, descriptors, global_features)
    if not equal_image_features(data_a.keypoints, data_b.keypoints):
        return False
    if not equal_image_features(data_a.descriptors, data_b.descriptors):
        return False
    if not equal_image_features(data_a.global_features, data_b.global_features):
        return False

    # compare matches
    if not equal_matches(data_a.matches, data_b.matches):
        return False

    # compare observations
    if not equal_observations(data_a.observations, data_b.observations):
        return False

    # compare points3d
    if not equal_points3d(data_a.points3d, data_b.points3d):
        return False
    return True
