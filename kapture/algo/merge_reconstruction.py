# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Merge kapture objects for reconstructions.
"""

from kapture.io.binary import array_to_file
from kapture.io.tar import TarCollection
import numpy as np
import os
import shutil
from typing import Dict, List, Union, Optional, Tuple, Type

import kapture
import kapture.io.features
from kapture.utils.logging import getLogger


def _merge_image_features(feature_class_type: Type[Union[kapture.Keypoints,
                                                         kapture.Descriptors,
                                                         kapture.GlobalFeatures]],
                          feature_type: str,
                          features_list: Union[List[Optional[kapture.Keypoints]],
                                               List[Optional[kapture.Descriptors]],
                                               List[Optional[kapture.GlobalFeatures]]],
                          features_paths: List[str],
                          output_path: str,
                          tar_handlers: List[TarCollection]
                          ) -> Union[kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures]:
    """
    Merge several features_list (keypoints, descriptors or global features_list) (of same type) in one.

    :param feature_class_type: the type of features_list
    :param feature_type: the type (name of the folder) of the features
    :param features_list: the list of values
    :param features_paths: the paths
    :param output_path: root path of the features to construct
    :param tar_handlers: collection of preloaded tar archives
    :return: merged features object of the corresponding type
    """
    assert len(features_list) > 0
    assert len(features_paths) == len(features_list)

    # find no none value
    val = [(i, d) for i, d in enumerate(features_list) if d is not None]
    assert len(val) > 0

    merged_features = val[0][1]
    for j, (i, features) in enumerate(val):
        assert isinstance(features, feature_class_type)
        assert features.type_name == merged_features.type_name
        assert features.dtype == merged_features.dtype
        assert features.dsize == merged_features.dsize
        if feature_class_type == kapture.Descriptors or feature_class_type == kapture.GlobalFeatures:
            assert not isinstance(features, kapture.Keypoints)  # IDE type check help
            assert not isinstance(merged_features, kapture.Keypoints)  # IDE type check help
            assert features.metric_type == merged_features.metric_type
        if feature_class_type == kapture.Descriptors:
            assert isinstance(features, kapture.Descriptors)  # IDE type check help
            assert isinstance(merged_features, kapture.Descriptors)  # IDE type check help
            assert features.keypoints_type == merged_features.keypoints_type
        for name in features:
            if j > 0 and name in merged_features:
                getLogger().warning(f'{name} was found multiple times.')
            else:
                merged_features.add(name)
                if output_path:
                    # TODO: uses kapture.io.features_list.get_image_features_dirpath()
                    in_path = kapture.io.features.get_features_fullpath(feature_class_type,
                                                                        feature_type,
                                                                        features_paths[i],
                                                                        name,
                                                                        tar_handlers[i])
                    out_path = kapture.io.features.get_features_fullpath(feature_class_type,
                                                                         feature_type,
                                                                         output_path,
                                                                         name)
                    if in_path != out_path:
                        # skip actual copy if file does not actually move.
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        if isinstance(in_path, str):
                            shutil.copy(in_path, out_path)
                        else:
                            # in_path is a tuple [str, TarHandler]
                            # keypoints are not stored in a file, have to read them to be able to copy them
                            array = in_path[1].get_array_from_tar(in_path[0], features.dtype, features.dsize)
                            array_to_file(out_path, array)
    return merged_features


def _merge_image_features_collection(feature_class_type: Type[Union[kapture.Keypoints,
                                                                    kapture.Descriptors,
                                                                    kapture.GlobalFeatures]],
                                     features_list: Union[List[Optional[Dict[str, kapture.Keypoints]]],
                                                          List[Optional[Dict[str, kapture.Descriptors]]],
                                                          List[Optional[Dict[str, kapture.GlobalFeatures]]]],
                                     features_paths: List[str],
                                     output_path: str,
                                     tar_handlers: List[TarCollection]
                                     ) -> Union[Dict[str, kapture.Keypoints],
                                                Dict[str, kapture.Descriptors],
                                                Dict[str, kapture.GlobalFeatures]]:
    assert len(features_list) > 0
    assert len(features_paths) == len(features_list)
    # get the union
    features_types = set().union(*[features.keys() for features in features_list if features is not None])
    if len(features_types) == 0:
        return {}

    out_collection = {}
    for features_type in features_types:
        image_features_list = [features[features_type] if features is not None and features_type in features else None
                               for features in features_list]
        image_features = _merge_image_features(feature_class_type, features_type,
                                               image_features_list,
                                               features_paths, output_path,
                                               tar_handlers)
        assert isinstance(image_features, feature_class_type)
        out_collection[features_type] = image_features
    return out_collection


def merge_keypoints(feature_type: str,
                    keypoints_list: List[Optional[kapture.Keypoints]],
                    keypoints_paths: List[str],
                    output_path: str,
                    tar_handlers: List[TarCollection]) -> kapture.Keypoints:
    """
    Merge several keypoints in one.

    :param keypoints_list: list of keypoints to merge
    :param keypoints_paths: keypoints files paths
    :param output_path: root path of the merged features files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged keypoints
    """
    keypoints = _merge_image_features(kapture.Keypoints, feature_type, keypoints_list, keypoints_paths,
                                      output_path, tar_handlers)
    assert isinstance(keypoints, kapture.Keypoints)
    return keypoints


def merge_keypoints_collections(keypoints_collections_list: List[Optional[Dict[str, kapture.Keypoints]]],
                                keypoints_paths: List[str],
                                output_path: str,
                                tar_handlers: List[TarCollection]) -> Dict[str, kapture.Keypoints]:
    """
    Merge several keypoints collections in one.

    :param keypoints_collections_list: list of keypoints collections to merge
    :param keypoints_paths: keypoints files paths
    :param output_path: root path of the merged features files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged keypoints collection
    """
    return _merge_image_features_collection(kapture.Keypoints, keypoints_collections_list,
                                            keypoints_paths, output_path, tar_handlers)


def merge_descriptors(feature_type: str,
                      descriptors_list: List[Optional[kapture.Descriptors]],
                      descriptors_paths: List[str], output_path: str,
                      tar_handlers: List[TarCollection]) -> kapture.Descriptors:
    """
    Merge several descriptors in one.

    :param descriptors_list: list of descriptors to merge
    :param descriptors_paths: descriptors files paths
    :param output_path: root path of the merged features files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged descriptors
    """
    descriptors = _merge_image_features(kapture.Descriptors, feature_type,
                                        descriptors_list, descriptors_paths, output_path, tar_handlers)
    assert isinstance(descriptors, kapture.Descriptors)
    return descriptors


def merge_descriptors_collections(descriptors_collections_list: List[Optional[Dict[str, kapture.Descriptors]]],
                                  descriptors_paths: List[str],
                                  output_path: str,
                                  tar_handlers: List[TarCollection]) -> Dict[str, kapture.Descriptors]:
    """
    Merge several descriptors collections in one.

    :param descriptors_collections_list: list of descriptors collections to merge
    :param descriptors_paths: descriptors files paths
    :param output_path: root path of the merged features files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged descriptors collections
    """
    return _merge_image_features_collection(kapture.Descriptors, descriptors_collections_list,
                                            descriptors_paths, output_path, tar_handlers)


def merge_global_features(global_features_list: List[Optional[kapture.GlobalFeatures]],
                          global_features_paths: List[str], output_path: str,
                          tar_handlers: List[TarCollection]) -> kapture.GlobalFeatures:
    """
    Merge several global features in one.

    :param global_features_list: list of global features to merge
    :param global_features_paths: global features files paths
    :param output_path: root path of the merged features files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged global features
    """
    features = _merge_image_features(kapture.GlobalFeatures, global_features_list, global_features_paths,
                                     output_path, tar_handlers)
    assert isinstance(features, kapture.GlobalFeatures)
    return features


def merge_global_features_collections(global_features_collections_list: List[Optional[Dict[str,
                                                                                           kapture.GlobalFeatures]]],
                                      global_features_paths: List[str],
                                      output_path: str,
                                      tar_handlers: List[TarCollection]) -> Dict[str, kapture.GlobalFeatures]:
    """
    Merge several global features collections in one.

    :param global_features_collections_list: list of global features collections to merge
    :param global_features_paths: global features files paths
    :param output_path: root path of the merged features files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged global features collection
    """
    return _merge_image_features_collection(kapture.GlobalFeatures, global_features_collections_list,
                                            global_features_paths, output_path, tar_handlers)


def merge_matches(keypoints_type: str,
                  matches_list: List[Optional[kapture.Matches]],
                  matches_paths: List[str],
                  output_path: str,
                  tar_handlers: List[TarCollection]) -> kapture.Matches:
    """
    Merge several matches lists in one.

    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param matches_list: list of matches to merge
    :param matches_paths: matches files paths
    :param output_path: root path of the merged matches files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged matches
    """
    assert len(matches_list) > 0
    assert len(matches_paths) == len(matches_list)

    merged_matches = kapture.Matches()
    for matches, matches_path, tar_handler in zip(matches_list, matches_paths, tar_handlers):
        if matches is None:
            continue
        for pair in matches:
            if pair in merged_matches:
                getLogger().warning(f'{pair} was found multiple times.')
            else:
                merged_matches.add(pair[0], pair[1])
                if output_path:
                    in_path = kapture.io.features.get_matches_fullpath(pair, keypoints_type, matches_path, tar_handler)
                    out_path = kapture.io.features.get_matches_fullpath(pair, keypoints_type, output_path)
                    if in_path != out_path:
                        # skip actual copy if file does not actually move.
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        if isinstance(in_path, str):
                            shutil.copy(in_path, out_path)
                        else:
                            # in_path is a tuple [str, TarHandler]
                            # keypoints are not stored in a file, have to read them to be able to copy them
                            array = kapture.io.features.image_matches_from_file(in_path)
                            kapture.io.features.image_matches_to_file(out_path, array)
    return merged_matches


def merge_matches_collections(matches_list: List[Optional[Dict[str,  kapture.Matches]]],
                              matches_paths: List[str],
                              output_path: str,
                              tar_handlers: List[TarCollection]) -> Dict[str,  kapture.Matches]:
    """
    Merge several matches collections in one.

    :param matches_list: list of matches collections to merge
    :param matches_paths: matches files paths
    :param output_path: root path of the merged matches files
    :param tar_handlers: collection of preloaded tar archives
    :return: merged matches collection
    """
    assert len(matches_list) > 0
    assert len(matches_paths) == len(matches_list)

    # get the union
    keypoints_types = set().union(*[matches.keys() for matches in matches_list if matches is not None])
    if len(keypoints_types) == 0:
        return {}

    out_collection = {}
    for keypoints_type in keypoints_types:
        kmatches_list = [matches[keypoints_type] if matches is not None and keypoints_type in matches else None
                         for matches in matches_list]
        merged_matches = merge_matches(keypoints_type,
                                       kmatches_list,
                                       matches_paths,
                                       output_path,
                                       tar_handlers)
        assert isinstance(merged_matches, kapture.Matches)
        out_collection[keypoints_type] = merged_matches
    return out_collection


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
            for point3d_idx, keypoints_type, (image_path, keypoint_idx) in kapture.flatten(observations):
                merged_observations.add(point3d_idx + point3d_offset, keypoints_type, image_path, keypoint_idx)
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
