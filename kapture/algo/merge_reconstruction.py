# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Merge kapture objects for reconstructions.
"""

import numpy as np
import os
import shutil
from typing import List, Union, Optional, Tuple, Type

import kapture
import kapture.io.features
from kapture.utils.logging import getLogger


def merge_image_features(feature_type: Type[Union[kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures]],
                         features_list: Union[List[Optional[kapture.Keypoints]],
                                              List[Optional[kapture.Descriptors]],
                                              List[Optional[kapture.GlobalFeatures]]],
                         features_paths: List[str],
                         output_path: str
                         ) -> Union[kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures]:
    """
    Merge several features_list (keypoints, descriptors or global features_list) (of same type) in one.

    :param feature_type: the type of features_list
    :param features_list: the list of values
    :param features_paths: the paths
    :param output_path: root path of the features to construct
    :return: merged features object of the corresponding type
    """
    assert len(features_list) > 0
    assert len(features_paths) == len(features_list)

    # find no none value
    val = [d for d in features_list if d is not None]
    assert len(val) > 0

    merged_features = feature_type(val[0].type_name, val[0].dtype, val[0].dsize)
    for features, features_path in zip(features_list, features_paths):
        if features is None:
            continue
        assert isinstance(features, feature_type)
        assert features.type_name == merged_features.type_name
        assert features.dtype == merged_features.dtype
        assert features.dsize == merged_features.dsize
        for name in features:
            if name in merged_features:
                getLogger().warning(f'{name} was found multiple times.')
            else:
                merged_features.add(name)
                if output_path:
                    # TODO: uses kapture.io.features_list.get_image_features_dirpath()
                    in_path = kapture.io.features.get_features_fullpath(feature_type, features_path, name)
                    out_path = kapture.io.features.get_features_fullpath(feature_type, output_path, name)
                    if in_path != out_path:
                        # skip actual copy if file does not actually move.
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        shutil.copy(in_path, out_path)
    return merged_features


def merge_keypoints(keypoints_list: List[Optional[kapture.Keypoints]],
                    keypoints_paths: List[str],
                    output_path: str) -> kapture.Keypoints:
    """
    Merge several keypoints in one.

    :param keypoints_list: list of keypoints to merge
    :param keypoints_paths: keypoints files paths
    :param output_path: root path of the merged features files
    :return: merged keypoints
    """
    keypoints = merge_image_features(kapture.Keypoints, keypoints_list, keypoints_paths, output_path)
    assert isinstance(keypoints, kapture.Keypoints)
    return keypoints


def merge_descriptors(descriptors_list: List[Optional[kapture.Descriptors]],
                      descriptors_paths: List[str], output_path: str) -> kapture.Descriptors:
    """
    Merge several descriptors in one.

    :param descriptors_list: list of descriptors to merge
    :param descriptors_paths: descriptors files paths
    :param output_path: root path of the merged features files
    :return: merged descriptors
    """
    descriptors = merge_image_features(kapture.Descriptors, descriptors_list, descriptors_paths, output_path)
    assert isinstance(descriptors, kapture.Descriptors)
    return descriptors


def merge_global_features(global_features_list: List[Optional[kapture.GlobalFeatures]],
                          global_features_paths: List[str], output_path: str) -> kapture.GlobalFeatures:
    """
    Merge several global features in one.

    :param global_features_list: list of global features to merge
    :param global_features_paths: global features files paths
    :param output_path: root path of the merged features files
    :return: merged global features
    """
    features = merge_image_features(kapture.GlobalFeatures, global_features_list, global_features_paths, output_path)
    assert isinstance(features, kapture.GlobalFeatures)
    return features


def merge_matches(matches_list: List[Optional[kapture.Matches]],
                  matches_paths: List[str],
                  output_path: str) -> kapture.Matches:
    """
    Merge several matches lists in one.

    :param matches_list: list of matches to merge
    :param matches_paths: matches files paths
    :param output_path: root path of the merged matches files
    :return: merged matches
    """
    assert len(matches_list) > 0
    assert len(matches_paths) == len(matches_list)

    merged_matches = kapture.Matches()
    for matches, matches_path in zip(matches_list, matches_paths):
        if matches is None:
            continue
        for pair in matches:
            if pair in merged_matches:
                getLogger().warning(f'{pair} was found multiple times.')
            else:
                merged_matches.add(pair[0], pair[1])
                if output_path:
                    in_path = kapture.io.features.get_matches_fullpath(pair, matches_path)
                    out_path = kapture.io.features.get_matches_fullpath(pair, output_path)
                    if in_path != out_path:
                        # skip actual copy if file does not actually move.
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        shutil.copy(in_path, out_path)
    return merged_matches


def merge_points3d_and_observations(pts3d_obs: List[Tuple[Optional[kapture.Points3d], Optional[kapture.Observations]]]
                                    ) -> Tuple[kapture.Points3d, kapture.Observations]:
    """
    Merge a list of points3d with their observations.

    :param pts3d_obs: list of points3d with observations to merge
    :return: merged points3d associated to observations
    """
    assert len(pts3d_obs) > 0
    merged_points3d = kapture.Points3d()
    merged_observations = kapture.Observations()
    point3d_offset = 0
    for points3d, observations in pts3d_obs:
        if points3d is None:
            continue
        merged_points3d = kapture.Points3d(np.vstack([merged_points3d, points3d]))
        if observations is not None:
            for point3d_idx, (image_path, keypoint_idx) in kapture.flatten(observations):
                merged_observations.add(point3d_idx + point3d_offset, image_path, keypoint_idx)
        point3d_offset += merged_points3d.shape[0]
    return merged_points3d, merged_observations


def merge_points3d(points3d_list: List[Optional[kapture.Points3d]]) -> kapture.Points3d:
    """
    Merge several points3d lists in one.

    :param points3d_list: list of points3d to merge
    :return: merged points3d
    """
    assert len(points3d_list) > 0
    merged_points3d = kapture.Points3d()
    for points3d in points3d_list:
        if points3d is None:
            continue
        merged_points3d = kapture.Points3d(np.vstack([merged_points3d, points3d]))
    return merged_points3d
