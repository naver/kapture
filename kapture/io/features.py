# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This files contains IO operations on Feature related data.
Features are processed data linked to one or multiple record data.
For example Keypoints are features related to 1 RecordCamera (image).
And Matches are features related to a pair of RecordCamera (2 images).
"""

from kapture.io.tar import TarCollection, TarHandler, list_files_in_tar, retrieve_tar_handler_from_collection
import numpy as np
import os.path as path
from typing import Tuple, Any, Dict, Type, Optional, Union, Iterable

import kapture
from kapture.io.records import RECORD_DATA_DIRNAME
from kapture.utils.paths import path_secure, populate_files_in_dirpath
from kapture.io.binary import array_from_file, array_to_file


# Feature files related functions ######################################################################################
FEATURES_DATA_DIRNAMES = {
    kapture.Keypoints: path.join('reconstruction', 'keypoints'),
    kapture.Descriptors: path.join('reconstruction', 'descriptors'),
    kapture.GlobalFeatures: path.join('reconstruction', 'global_features'),
    kapture.Matches: path.join('reconstruction', 'matches'),
    kapture.RecordsBase: RECORD_DATA_DIRNAME,  # prefer kapture.RecordsCamera, Lidar, ...
}

FEATURE_FILE_EXTENSION = {
    kapture.Keypoints: '.kpt',
    kapture.Descriptors: '.desc',
    kapture.GlobalFeatures: '.gfeat',
    kapture.Matches: '.matches',
}

FEATURE_PAIR_PATH_SEPARATOR = {
    kapture.Matches: '.overlapping'
}


def guess_feature_name_from_path(feature_path: str) -> str:
    """
    try to find keypoints_type, descriptors_type, global_features_type from their full path

    :param feature_path: examples,  dataset/local_features/r2d2/keypoints ;
                                    dataset/local_features/r2d2 ;
                                    kapture_path/reconstruction/keypoints/r2d2
                                    dataset/global_features/apgem/global_features ;
                                    dataset/global_features/apgem ;
                                    kapture_path/reconstruction/global_features/apgem ;
    """
    feature_path_c = path.abspath(feature_path).replace('\\', '/').rstrip('/')
    feature_path_split = feature_path_c.split('/')
    feature_name = None
    # if path given doesn't end with the kapture name, assume the last bit is the feature name
    # ex dataset/local_features/r2d2
    # ex kapture_path/reconstruction/keypoints/r2d2
    if feature_path_split[-1] not in ['keypoints', 'descriptors', 'global_features', 'matches']:
        feature_name = feature_path_split[-1]
    else:
        # path given look like it follows the kapture-localization recommendation
        # search for the local_features; global_features keywords
        for parent_folder_name in ['local_features', 'global_features']:
            if feature_name is not None:
                break
            try:
                indices = [i for i, x in enumerate(feature_path_split) if x == parent_folder_name]
                if len(indices) > 0:
                    # from last occurence to first
                    for index in reversed(indices):
                        # ignore if keyword is at the very end of the sequence
                        # would happen for dataset/global_features/apgem/global_features
                        if index + 1 < len(feature_path_split):
                            feature_name = feature_path_split[index + 1]
                            break
            except Exception:
                continue
    if feature_name is None:
        raise ValueError(f'failed to guess feature name from path {feature_path}')
    return feature_name


# get file path for binary files in kapture ############################################################################
def get_features_fullpath(
    data_type: Any,
    feature_type: str,
    kapture_dirpath: str = '',
    image_filename: Optional[str] = None,
    tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Union[str, Tuple[Optional[str], TarHandler]]:
    """
    Returns full path to subdirectory containing the binary files of the given type.
            Optionally, can give a the image file name, and add the feature file name (with proper extension).

    :param data_type:
    :param feature_type: name of the feature
    :param kapture_dirpath: input path to kapture directory.
    :param image_filename: optional input image filename (id).
    :param tar_handler: collection of preloaded tar archives
    :return: Feature full path
    """
    assert feature_type is not None
    tar_local_handler = retrieve_tar_handler_from_collection(data_type, feature_type, tar_handler)
    if tar_local_handler is None:
        subdir = FEATURES_DATA_DIRNAMES[data_type]
        feature_filename = image_filename + FEATURE_FILE_EXTENSION[data_type] if image_filename else ''
        return path_secure(path.join(kapture_dirpath, subdir, feature_type, feature_filename))
    else:
        feature_filename = image_filename + FEATURE_FILE_EXTENSION[data_type] if image_filename else None
        return feature_filename, tar_local_handler


def features_to_filepaths(
    kapture_data: Union[kapture.Keypoints,
                        kapture.Descriptors,
                        kapture.GlobalFeatures],
    feature_type: str,
    kapture_dirpath: str = '',
    tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Dict[str, Union[str, Tuple[str, TarHandler]]]:
    """
    Returns a dict mapping image_id to path to feature file.
        Eg.{image_id: full/path/to/feature.jpg.kpt}
        or
        {image_id: (feature.jpg.kpt, ref_to_tar)}

    :param kapture_data: input kapture data.
    :param feature_type: name of the feature
    :param kapture_dirpath: input path to kapture directory.
    :param tar_handler: collection of preloaded tar archives
    :return: image id to image file path dictionary
    """
    data_type = type(kapture_data)
    return {
        image_filename: get_features_fullpath(data_type, feature_type, kapture_dirpath, image_filename, tar_handler)
        for image_filename in kapture_data
    }


def image_ids_from_feature_tar(kapture_type: Type[Union[kapture.Keypoints,
                                                        kapture.Descriptors,
                                                        kapture.GlobalFeatures]],
                               tar_handler: TarHandler) -> Iterable[str]:
    """
     Populate feature files in tar and returns their corresponding image_filename

    :param kapture_type: kapture class type.
    :param tar_handler: opened tar reference
    :return: image file paths
    """
    feature_filenames = list_files_in_tar(tar_handler, FEATURE_FILE_EXTENSION[kapture_type])
    image_filenames = (feature_filename[:-len(FEATURE_FILE_EXTENSION[kapture_type])]
                       for feature_filename in feature_filenames)
    return image_filenames


def image_ids_from_feature_dirpath(
        kapture_type: Type[Union[kapture.Keypoints,
                                 kapture.Descriptors,
                                 kapture.GlobalFeatures]],
        feature_type: str,
        kapture_dirpath: str = '') -> Iterable[str]:
    """
    Populate feature files and returns their corresponding image_filename.

    :param kapture_type: kapture class type.
    :param feature_type: name of the feature
    :param kapture_dirpath: root path of kapture
    :return: image file paths
    """
    feature_dirpath = get_features_fullpath(kapture_type, feature_type, kapture_dirpath)
    # filter only files with proper extension and
    feature_filenames = populate_files_in_dirpath(feature_dirpath, FEATURE_FILE_EXTENSION[kapture_type])
    # remove file extensions to retrieve image name.
    image_filenames = (feature_filename[:-len(FEATURE_FILE_EXTENSION[kapture_type])]
                       for feature_filename in feature_filenames)
    return image_filenames


def features_check_dir(
        kapture_data: Union[kapture.Keypoints,
                            kapture.Descriptors,
                            kapture.GlobalFeatures,
                            kapture.Matches],
        feature_type: str,
        kapture_dirpath: str,
        tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> bool:
    """
    Makes sure all files actually exists.

    :param kapture_data: input kapture data
    :param feature_type: name of the feature
    :param kapture_dirpath: root path of kapture
    :param tar_handler: collection of preloaded tar archives
    :return: True only if all exist, false otherwise
    """
    data_type = type(kapture_data)
    tar_local_handler = retrieve_tar_handler_from_collection(data_type, feature_type, tar_handler)
    file_list = features_to_filepaths(kapture_data, feature_type, kapture_dirpath, tar_local_handler).values()
    if tar_local_handler is None:
        all_files_exists = all(path.exists(feature_filepath) for feature_filepath in file_list)
    else:
        all_files_in_tar = set(list_files_in_tar(tar_local_handler, FEATURE_FILE_EXTENSION[data_type]))
        all_files_exists = all(feature_filepath[0] in all_files_in_tar for feature_filepath in file_list)
    return all_files_exists


# image_keypoints ######################################################################################################
def image_keypoints_from_file(filepath: Union[str, Tuple[str, TarHandler]], dtype: Type, dsize: int) -> np.ndarray:
    """
    Read the image keypoints

    :param filepath: path to the file
    :param dtype: data type
    :param dsize: number of data per keypoint
    :return: the image keypoints
    """
    if isinstance(filepath, str):
        return array_from_file(filepath, dtype, dsize)
    else:
        return filepath[1].get_array_from_tar(filepath[0], dtype, dsize)


def image_keypoints_to_file(filepath: Union[str, Tuple[str, TarHandler]], image_keypoints: np.ndarray) -> None:
    """
    Writes the image keypoints to file

    :param filepath: file path
    :param image_keypoints: image keypoints
    """
    if isinstance(filepath, str):
        array_to_file(filepath, image_keypoints)
    else:
        filepath[1].add_array_to_tar(filepath[0], image_keypoints)


def get_keypoints_fullpath(keypoints_type: str, kapture_dirpath: str, image_filename: Optional[str] = None,
                           tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Union[str, Tuple[Optional[str], TarHandler]]:
    """
    Computes the full path of the keypoints file

    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: top kapture directory path
    :param image_filename: image file name
    :param tar_handler: collection of preloaded tar archives
    :return: full path of the keypoints file
    """
    return get_features_fullpath(kapture.Keypoints, keypoints_type, kapture_dirpath, image_filename, tar_handler)


def keypoints_to_filepaths(keypoints: kapture.Keypoints, keypoints_type: str, kapture_dirpath: str,
                           tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Dict[str, Union[str, Tuple[str, TarHandler]]]:
    """
    Computes keypoints files paths

    :param keypoints: keypoints
    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: keypoint to keypoint file dictionary
    """
    return features_to_filepaths(keypoints, keypoints_type, kapture_dirpath, tar_handler)


def keypoints_check_dir(keypoints: kapture.Keypoints, keypoints_type: str, kapture_dirpath: str,
                        tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> bool:
    """
    Checks that all keypoints file exist.

    :param keypoints: keypoints
    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: True if they all exist, false otherwise.
    """
    return features_check_dir(keypoints, keypoints_type, kapture_dirpath, tar_handler)


# image_descriptors ####################################################################################################
def image_descriptors_from_file(filepath: Union[str, Tuple[str, TarHandler]], dtype: Type, dsize: int) -> np.ndarray:
    """
    Read the image descriptors

    :param filepath: path to the file
    :param dtype: data type
    :param dsize: number of data per keypoint
    :return: the image descriptors
    """
    if isinstance(filepath, str):
        return array_from_file(filepath, dtype, dsize)
    else:
        return filepath[1].get_array_from_tar(filepath[0], dtype, dsize)


def image_descriptors_to_file(filepath: Union[str, Tuple[str, TarHandler]], image_descriptors: np.ndarray) -> None:
    """
    Writes the image descriptors to file

    :param filepath: file path
    :param image_descriptors: image descriptors
    """
    if isinstance(filepath, str):
        array_to_file(filepath, image_descriptors)
    else:
        filepath[1].add_array_to_tar(filepath[0], image_descriptors)


def get_descriptors_fullpath(descriptors_type: str, kapture_dirpath: str, image_filename: Optional[str] = None,
                             tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Union[str, Tuple[Optional[str], TarHandler]]:
    """
    Computes the full path of the descriptors file

    :param descriptors_type: type of descriptors to export, name of the descriptors subfolder
    :param kapture_dirpath: top kapture directory path
    :param image_filename: image file name
    :param tar_handler: collection of preloaded tar archives
    :return: full path of the descriptors file
    """
    return get_features_fullpath(kapture.Descriptors, descriptors_type, kapture_dirpath, image_filename, tar_handler)


def descriptors_to_filepaths(descriptors: kapture.Descriptors,
                             descriptors_type: str,
                             kapture_dirpath: str,
                             tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Dict[str, Union[str, Tuple[str, TarHandler]]]:
    """
    Computes descriptors files paths

    :param descriptors: descriptors
    :param descriptors_type: type of descriptors to export, name of the descriptors subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: descriptors to descriptors file dictionary
    """
    return features_to_filepaths(descriptors, descriptors_type, kapture_dirpath, tar_handler)


def descriptors_check_dir(descriptors: kapture.Descriptors, descriptors_type: str, kapture_dirpath: str,
                          tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> bool:
    """
    Checks that all descriptors file exist.

    :param descriptors: descriptors
    :param descriptors_type: type of descriptors to export, name of the descriptors subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: True if they all exist, false otherwise.
    """
    return features_check_dir(descriptors, descriptors_type, kapture_dirpath, tar_handler)


# global_features ######################################################################################################
def image_global_features_from_file(filepath: Union[str, Tuple[str, TarHandler]], dtype: Type, dsize: int)\
        -> np.ndarray:
    """
    Read the image global features

    :param filepath: path to the file
    :param dtype: data type
    :param dsize: number of data per keypoint
    :return: the global features
    """
    if isinstance(filepath, str):
        return array_from_file(filepath, dtype, dsize)
    else:
        return filepath[1].get_array_from_tar(filepath[0], dtype, dsize)


def image_global_features_to_file(filepath: Union[str, Tuple[str, TarHandler]],
                                  image_global_descriptor: np.ndarray) -> None:
    """
    Writes the image global features to file

    :param filepath: file path
    :param image_global_descriptor: image global features
    """
    if isinstance(filepath, str):
        array_to_file(filepath, image_global_descriptor)
    else:
        filepath[1].add_array_to_tar(filepath[0], image_global_descriptor)


def get_global_features_fullpath(global_features_type: str,
                                 kapture_dirpath: str,
                                 image_filename: Optional[str] = None,
                                 tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Union[str, Tuple[Optional[str], TarHandler]]:
    """
    Computes the full path of the global features file

    :param kapture_dirpath: top kapture directory path
    :param global_features_type: type of global_features, name of the global_features subfolder
    :param image_filename: image file name
    :param tar_handler: collection of preloaded tar archives
    :return: full path of the global features file
    """
    return get_features_fullpath(kapture.GlobalFeatures, global_features_type, kapture_dirpath,
                                 image_filename, tar_handler)


def global_features_to_filepaths(global_features: kapture.GlobalFeatures,
                                 global_features_type: str,
                                 kapture_dirpath: str,
                                 tar_handler: Optional[Union[TarCollection, TarHandler]] = None) \
        -> Dict[str, Union[str, Tuple[str, TarHandler]]]:
    """
    Computes global features files paths

    :param global_features: global features
    :param global_features_type: type of global_features, name of the global_features subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: global features to global features file dictionary
    """
    return features_to_filepaths(global_features, global_features_type, kapture_dirpath, tar_handler)


def global_features_check_dir(global_features: kapture.GlobalFeatures,
                              global_features_type: str,
                              kapture_dirpath: str,
                              tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> bool:
    """
    Checks that all keypoints file exist.

    :param global_features: global features
    :param global_features_type: type of global_features, name of the global_features subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: True if they all exist, false otherwise.
    """
    return features_check_dir(global_features, global_features_type, kapture_dirpath, tar_handler)


# matches ##############################################################################################################
def image_matches_from_file(filepath: Union[str, Tuple[str, TarHandler]]) -> np.ndarray:
    """
    Read the image matches
     [keypoint_id_image1, keypoint_id_image2, score]

    :param filepath: path to the file
    :return: matches
    """
    if isinstance(filepath, str):
        return array_from_file(filepath, dtype=np.float64, dsize=3)
    else:
        return filepath[1].get_array_from_tar(filepath[0], dtype=np.float64, dsize=3)


def image_matches_to_file(filepath: Union[str, Tuple[str, TarHandler]], image_matches: np.ndarray) -> None:
    """
    Writes the image matches to file

    :param filepath: file path
    :param image_matches: image matches
    """
    assert image_matches.dtype == np.float64
    assert image_matches.shape[1] == 3
    if isinstance(filepath, str):
        array_to_file(filepath, image_matches)
    else:
        filepath[1].add_array_to_tar(filepath[0], image_matches)


def get_matches_fullpath(
        image_filename_pair: Optional[Tuple[str, str]] = None,
        keypoints_type: str = '',
        kapture_dirpath: str = '',
        tar_handler: Optional[Union[TarCollection, TarHandler]] = None)  \
        -> Union[str, Tuple[Optional[str], TarHandler]]:
    """
    Computes the full path of the matches file between two images

    :param image_filename_pair: image file names
    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: full path of the matches file
    """
    if image_filename_pair is not None:
        filename = path_secure(path.join(image_filename_pair[0] + FEATURE_PAIR_PATH_SEPARATOR[kapture.Matches],
                                         image_filename_pair[1]))
    else:
        filename = None
    return get_features_fullpath(kapture.Matches, keypoints_type, kapture_dirpath, filename, tar_handler)


def matches_to_filepaths(matches: kapture.Matches,
                         keypoints_type: str,
                         kapture_dirpath: str = '',
                         tar_handler: Optional[Union[TarCollection, TarHandler]] = None)  \
        -> Dict[Tuple[str, str], Union[str, Tuple[str, TarHandler]]]:
    """
    Computes matches files paths

    :param matches: matches
    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: file to file matches to matches file dictionary
    """
    return {
        image_filename_pair: get_matches_fullpath(image_filename_pair, keypoints_type, kapture_dirpath, tar_handler)
        for image_filename_pair in matches
    }


def _matches_filenames_remove_extensions_and_cut(matches_filenames: Iterable[str]) -> Iterable[Tuple[str, str]]:
    # remove the extensions and cut
    matching_pairs = (
        path_secure(matches_filename)[:-len(FEATURE_FILE_EXTENSION[kapture.Matches])
                                      ].split(FEATURE_PAIR_PATH_SEPARATOR[kapture.Matches] + '/')
        for matches_filename in matches_filenames
    )
    matching_pairs = ((matches[0], matches[1]) for matches in matching_pairs if len(matches) == 2)
    return matching_pairs


def matching_pairs_from_tar(tar_handler: TarHandler) -> Iterable[Tuple[str, str]]:
    """
    Read and build Matches from a tar file.
    """
    matches_filenames = list_files_in_tar(tar_handler, FEATURE_FILE_EXTENSION[kapture.Matches])
    return _matches_filenames_remove_extensions_and_cut(matches_filenames)


def matching_pairs_from_dirpath(keypoints_type: str, kapture_dirpath: str) -> Iterable[Tuple[str, str]]:
    """
    Read and build Matches from kapture directory tree.
    """
    matches_dirpath = get_matches_fullpath(None, keypoints_type, kapture_dirpath)
    # list all files there is
    # filter only match files (the ones endings with .matches)
    matches_filenames = populate_files_in_dirpath(matches_dirpath, FEATURE_FILE_EXTENSION[kapture.Matches])
    return _matches_filenames_remove_extensions_and_cut(matches_filenames)


def matches_check_dir(matches: kapture.Matches, keypoints_type: str, kapture_dirpath: str,
                      tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> bool:
    """
    Checks that all matches file exist.

    :param matches: matches
    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: top kapture directory path
    :param tar_handler: collection of preloaded tar archives
    :return: True if they all exist, false otherwise.
    """
    tar_local_handler = retrieve_tar_handler_from_collection(kapture.Matches, keypoints_type, tar_handler)
    file_list = (get_matches_fullpath(pair, keypoints_type, kapture_dirpath, tar_local_handler) for pair in matches)
    if tar_local_handler is None:
        all_files_exists = all(path.exists(filepath) for filepath in file_list)
    else:
        all_files_in_tar = set(list_files_in_tar(tar_local_handler, FEATURE_FILE_EXTENSION[kapture.Matches]))
        all_files_exists = all(feature_filepath[0] in all_files_in_tar for feature_filepath in file_list)
    return all_files_exists
