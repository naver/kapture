# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Operations on kapture pose objects
"""

import math
import numpy as np
import quaternion
from typing import Tuple, Union, List

import kapture


def pose_transform_distance(pose_a: kapture.PoseTransform, pose_b: kapture.PoseTransform) -> Tuple[float, float]:
    """
    get translation and rotation distance between two PoseTransform

    :return: (position_distance, rotation_distance in rad), can be nan is case of invalid comparison
    """
    # handle NoneType with try expect blocks
    try:
        translation_distance = np.linalg.norm(pose_a.t - pose_b.t)
    except TypeError:
        translation_distance = math.nan

    try:
        rotation_distance = quaternion.rotation_intrinsic_distance(pose_a.r, pose_b.r)
    except TypeError:
        rotation_distance = math.nan
    return translation_distance, rotation_distance


def world_pose_transform_distance(pose_a: kapture.PoseTransform, pose_b: kapture.PoseTransform) -> Tuple[float, float]:
    """
    get position and rotation error between two PoseTransform
    pose_a and pose_b should be world to device

    :return: (position_distance, rotation_distance in deg), can be nan is case of invalid comparison
    """
    if pose_a.r is None and pose_a.t is None:
        return math.nan, math.nan
    elif pose_a.r is None:
        device_to_world_a = kapture.PoseTransform(r=None, t=-pose_a.t)
    elif pose_a.t is None:
        device_to_world_a = kapture.PoseTransform(r=pose_a.r.inverse(), t=None)
    else:
        device_to_world_a = pose_a.inverse()

    if pose_b.r is None and pose_b.t is None:
        return math.nan, math.nan
    elif pose_b.r is None:
        device_to_world_b = kapture.PoseTransform(r=None, t=-pose_b.t)
    elif pose_b.t is None:
        device_to_world_b = kapture.PoseTransform(r=pose_b.r.inverse(), t=None)
    else:
        device_to_world_b = pose_b.inverse()

    pose_error = pose_transform_distance(device_to_world_a, device_to_world_b)
    return pose_error[0], np.rad2deg(pose_error[1])


def average_quaternion(big_q: np.ndarray) -> np.ndarray:
    """
    Computes the Chordal L2-Mean using quaternions.
    Ported from Tolga Birdal's implementation
    https://github.com/tolgabirdal/averaging_quaternions/blob/master/avg_quaternion_markley.m (MIT)

    :param big_q: Q (or big_q in python) is an (M,4) ndarray of quaternions
    :return: float array representing the average quaternion
    """
    # Form the symmetric accumulator matrix
    big_a = np.zeros((4, 4))
    big_m = big_q.shape[0]
    for i in range(big_m):
        q = big_q[i, :]
        big_a += np.outer(q, q)  # rank 1 update
    # scale
    big_a /= big_m
    # Get the eigen vector corresponding to largest eigen value
    return np.linalg.eigh(big_a)[1][:, -1]


def average_pose_transform(poses: List[kapture.PoseTransform]) -> kapture.PoseTransform:
    """
    average a list of poses with equal weights

    :param poses: list of poses to average
    :return: average PoseTransform
    """
    assert isinstance(poses, list)
    assert len(poses) > 0

    # handle NoneType with try expect blocks
    try:
        translation = np.sum(tuple([pose.t for pose in poses]), axis=0) / len(poses)
    except TypeError:
        translation = None

    try:
        rotation = average_quaternion(np.vstack(tuple([quaternion.as_float_array(pose.r) for pose in poses])))
    except TypeError:
        rotation = None

    return kapture.PoseTransform(r=rotation, t=translation)


def average_quaternion_weighted(big_q: np.ndarray, weights: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Averaging Quaternions.
    Ported from Tolga Birdal's implementation
    https://github.com/tolgabirdal/averaging_quaternions/blob/master/wavg_quaternion_markley.m (MIT)

    :param big_q: Q is an (M,4) ndarray of quaternions
    :param weights: a (M,) vector
    :return: float array representing the average quaternion
    """
    # Form the symmetric accumulator matrix
    big_a = np.zeros((4, 4))
    big_m = big_q.shape[0]
    w_sum = 0

    for i in range(big_m):
        q = big_q[i, :]
        w_i = weights[i]
        big_a += w_i * (np.outer(q, q))  # rank 1 update
        w_sum += w_i
    # scale
    big_a /= w_sum
    # Get the eigen vector corresponding to largest eigen value
    return np.linalg.eigh(big_a)[1][:, -1]


def average_pose_transform_weighted(poses: List[kapture.PoseTransform],
                                    weights: Union[List[float], np.ndarray]) -> kapture.PoseTransform:
    """
    average a list of poses with any weights

    :param poses: list of poses to average
    :param weights: a (len(poses),) vector
    :return: average PoseTransform
    """

    assert isinstance(poses, list)
    assert len(poses) > 0

    assert isinstance(weights, list)
    assert len(weights) == len(poses)

    weight_sum = np.sum(weights)

    # handle NoneType with try expect blocks
    try:
        translation = np.sum(tuple([weights[i] * pose.t for i, pose in enumerate(poses)]), axis=0) / weight_sum
    except TypeError:
        translation = None

    try:
        rotation = average_quaternion_weighted(np.vstack(tuple([quaternion.as_float_array(pose.r) for pose in poses])),
                                               weights)
    except TypeError:
        rotation = None

    return kapture.PoseTransform(r=rotation, t=translation)
