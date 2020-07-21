# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Evaluation with kapture objects
"""

import math
from typing import Union, List, Tuple, Set

import kapture
from .pose_operations import world_pose_transform_distance


def evaluate_error_absolute(poses_to_test: List[Tuple[str, kapture.PoseTransform]],
                            poses_ground_truth: List[Tuple[str, kapture.PoseTransform]]
                            ) -> List[Tuple[str, float, float]]:
    """
    Evaluate the absolute error for poses to a ground truth.

    :param poses_to_test: poses to test
    :param poses_ground_truth: reference poses
    :return: list of error evaluation
    """
    poses_ground_truth_as_dict = {name: pose for name, pose in poses_ground_truth}
    result = [(name,) + world_pose_transform_distance(pose, poses_ground_truth_as_dict[name])
              for (name, pose) in poses_to_test]
    return result


def get_poses(k_data: kapture.Kapture,
              image_set: Union[Set[str], List[str]]) -> List[Tuple[str, kapture.PoseTransform]]:
    """
    Computes the poses for a set of images within a kapture.

    :param k_data: the kapture
    :param image_set: set of image names
    :return: list of (image name,pose)
    """
    assert k_data.trajectories is not None
    if isinstance(image_set, list):
        image_set = set(image_set)
    assert isinstance(image_set, set)
    assert isinstance(k_data, kapture.Kapture)

    # apply rigs to trajectories
    if k_data.rigs is not None:
        trajectories = kapture.rigs_remove(k_data.trajectories, k_data.rigs)
    else:
        trajectories = k_data.trajectories

    poses = []
    for timestamp, device_id, filename in kapture.flatten(k_data.records_camera, is_sorted=True):
        if filename in image_set and (timestamp, device_id) in trajectories:
            pose = trajectories[(timestamp, device_id)]
            poses.append((filename, pose))
    return poses


# def get_imagename_intersection(kdata_list: List[kapture.Kapture]) -> Set[str]:
#     def get_image_set(k_data: kapture.Kapture):
#         return set(image_name for _, _, image_name in kapture.flatten(k_data.records_camera)) \
#             if k_data.records_camera is not None else set()
#     assert isinstance(kdata_list, list)
#     assert len(kdata_list) > 0
#     return set.intersection(*[get_image_set(k_data) for k_data in kdata_list])


def evaluate(k_data: kapture.Kapture,
             k_data_gt: kapture.Kapture,
             image_set: Union[Set[str], List[str]]) -> List[Tuple[str, float, float]]:
    """
    Evaluate the pose found for images in a kapture with a reference kapture.

    :param k_data: the kapture to test
    :param k_data_gt: the reference kapture
    :param image_set: list of image names
    :return: list of image pose evaluation
    """
    if isinstance(image_set, list):
        image_set = set(image_set)
    assert isinstance(image_set, set)
    assert(len(image_set) > 0)
    assert isinstance(k_data, kapture.Kapture)
    assert isinstance(k_data_gt, kapture.Kapture)

    poses_to_test = get_poses(k_data, image_set)
    poses_gt = get_poses(k_data_gt, image_set)

    evaluated = evaluate_error_absolute(poses_to_test, poses_gt)
    localized_images = {name for name, position_error, rotation_error in evaluated}
    missing_images = [name for name in image_set if name not in localized_images]
    for name in missing_images:
        evaluated.append((name, math.nan, math.nan))
    return sorted(evaluated)


def fill_bins(results: List[Tuple[str, float, float]],
              bins: List[Tuple[float, float]]
              ) -> List[Tuple[float, float, int]]:
    """
    Fill a bin with the number of images within position thresholds.

    :param results: list of error evaluation (image name, translation error, rotation error)
    :param bins: list of translation and rotation thresholds
    :return: number of images in every pair of (translation,rotation) error
    """
    assert isinstance(results, list)
    assert isinstance(bins, list)

    all_positions = [(translation_error, rotation_error) for name, translation_error, rotation_error in results]
    filled_bins = []
    for a_bin in bins:
        trans_threshold = a_bin[0]
        rot_threshold = a_bin[1]
        number_of_images_in_bin = 0
        for translation_error, rotation_error in all_positions:
            if (math.isnan(rot_threshold) or rot_threshold < 0) and translation_error <= trans_threshold:
                number_of_images_in_bin += 1
            elif translation_error <= trans_threshold and rotation_error <= rot_threshold:
                number_of_images_in_bin += 1
        filled_bins.append((trans_threshold, rot_threshold, number_of_images_in_bin))
    return filled_bins
