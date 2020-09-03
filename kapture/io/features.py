# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This files contains IO operations on Feature related data.
Features are processed data linked to one or multiple record data.
For example Keypoints are features related to 1 RecordCamera (image).
And Matches are features related to a pair of RecordCamera (2 images).
"""

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


# get file path for binary files in kapture ############################################################################
def get_features_fullpath(
        data_type: Any,
        kapture_dirpath: str = '',
        image_filename: Optional[str] = None) -> str:
    """
    Returns full path to subdirectory containing the binary files of the given type.
            Optionally, can give a the image file name, and add the feature file name (with proper extension).

    :param data_type:
    :param kapture_dirpath: input path to kapture directory.
    :param image_filename: optional input image filename (id).
    :return: Feature full path
    """
    subdir = FEATURES_DATA_DIRNAMES[data_type]
    feature_filename = image_filename + FEATURE_FILE_EXTENSION[data_type] if image_filename else ''
    return path_secure(path.join(kapture_dirpath, subdir, feature_filename))


def features_to_filepaths(
        kapture_data: Union[kapture.RecordsCamera,
                            kapture.Keypoints,
                            kapture.Descriptors,
                            kapture.GlobalFeatures],
        kapture_dirpath: str = '') -> Dict[str, str]:
    """
    Returns a dict mapping image_id to path to feature file.
        Eg.{image_id: full/path/to/feature.jpg.kpt}

    :param kapture_data: input kapture data.
    :param kapture_dirpath: input path to kapture directory.
    :return: image id to image file path dictionary
    """
    data_type = type(kapture_data)
    return {
        image_filename: get_features_fullpath(data_type, kapture_dirpath, image_filename)
        for image_filename in kapture_data
    }


def image_ids_from_feature_dirpath(
        kapture_type: Type[Union[kapture.Keypoints,
                                 kapture.Descriptors,
                                 kapture.GlobalFeatures,
                                 kapture.Matches]],
        kapture_dirpath: str = '') -> Dict[str, str]:
    """
    Populate feature files and returns their corresponding image_filename.

    :param kapture_type:
    :param kapture_dirpath:
    :return: image file path
    """
    feature_dirpath = get_features_fullpath(kapture_type, kapture_dirpath)
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
        kapture_dirpath: str) -> bool:
    """
    Makes sure all files actually exists.

    :param kapture_data:
    :param kapture_dirpath:
    :return: True only if all exist, false otherwise
    """
    file_list = features_to_filepaths(kapture_data, kapture_dirpath).values()
    all_files_exists = all(path.exists(feature_filepath) for feature_filepath in file_list)
    return all_files_exists


# image_keypoints ######################################################################################################
def image_keypoints_from_file(filepath: str, dtype: Type, dsize: int) -> np.array:
    """
    Read the image keypoints

    :param filepath: path to the file
    :param dtype: data type
    :param dsize: number of data per keypoint
    :return: the image keypoints
    """
    return array_from_file(filepath, dtype, dsize)


def image_keypoints_to_file(filepath: str, image_keypoints: np.array) -> None:
    """
    Writes the image keypoints to file

    :param filepath: file path
    :param image_keypoints: image keypoints
    """
    array_to_file(filepath, image_keypoints)


def get_keypoints_fullpath(kapture_dirpath: str, image_filename: Optional[str] = None) -> str:
    """
    Computes the full path of the keypoints file

    :param kapture_dirpath: top kapture directory path
    :param image_filename: image file name
    :return: full path of the keypoints file
    """
    return get_features_fullpath(kapture.Keypoints, kapture_dirpath, image_filename)


def keypoints_to_filepaths(keypoints: kapture.Keypoints, kapture_dirpath: str) -> Dict[str, str]:
    """
    Computes keypoints files paths

    :param keypoints: keypoints
    :param kapture_dirpath: top kapture directory path
    :return: keypoint to keypoint file dictionary
    """
    return features_to_filepaths(keypoints, kapture_dirpath)


def keypoints_check_dir(keypoints: kapture.Keypoints, kapture_dirpath: str) -> bool:
    """
    Checks that all keypoints file exist.

    :param keypoints: keypoints
    :param kapture_dirpath: top kapture directory path
    :return: True if they all exist, false otherwise.
    """
    return features_check_dir(keypoints, kapture_dirpath)


# image_descriptors ####################################################################################################
def image_descriptors_from_file(filepath: str, dtype: Type, dsize: int) -> np.array:
    """
    Read the image descriptors

    :param filepath: path to the file
    :param dtype: data type
    :param dsize: number of data per keypoint
    :return: the image descriptors
    """
    return array_from_file(filepath, dtype, dsize)


def image_descriptors_to_file(filepath: str, image_descriptors: np.array) -> None:
    """
    Writes the image descriptors to file

    :param filepath: file path
    :param image_descriptors: image descriptors
    """
    array_to_file(filepath, image_descriptors)


def get_descriptors_fullpath(kapture_dirpath: str, image_filename: Optional[str] = None) -> str:
    """
    Computes the full path of the descriptors file

    :param kapture_dirpath: top kapture directory path
    :param image_filename: image file name
    :return: full path of the descriptors file
    """
    return get_features_fullpath(kapture.Descriptors, kapture_dirpath, image_filename)


def descriptors_to_filepaths(descriptors: kapture.Descriptors, kapture_dirpath: str) -> Dict[str, str]:
    """
    Computes descriptors files paths

    :param descriptors: descriptors
    :param kapture_dirpath: top kapture directory path
    :return: descriptors to descriptors file dictionary
    """
    return features_to_filepaths(descriptors, kapture_dirpath)


def descriptors_check_dir(descriptors: kapture.Descriptors, kapture_dirpath: str) -> bool:
    """
    Checks that all descriptors file exist.

    :param descriptors: descriptors
    :param kapture_dirpath: top kapture directory path
    :return: True if they all exist, false otherwise.
    """
    return features_check_dir(descriptors, kapture_dirpath)


# global_features ######################################################################################################
def image_global_features_from_file(filepath: str, dtype: Type, dsize: int) -> np.array:
    """
    Read the image global features

    :param filepath: path to the file
    :param dtype: data type
    :param dsize: number of data per keypoint
    :return: the global features
    """
    return array_from_file(filepath, dtype, dsize)


def image_global_features_to_file(filepath: str, image_global_descriptor: np.array) -> None:
    """
    Writes the image global features to file

    :param filepath: file path
    :param image_global_descriptor: image global features
    """
    array_to_file(filepath, image_global_descriptor)


def get_global_features_fullpath(kapture_dirpath: str, image_filename: Optional[str] = None) -> str:
    """
    Computes the full path of the global features file

    :param kapture_dirpath: top kapture directory path
    :param image_filename: image file name
    :return: full path of the global features file
    """
    return get_features_fullpath(kapture.GlobalFeatures, kapture_dirpath, image_filename)


def global_features_to_filepaths(global_features: kapture.GlobalFeatures, kapture_dirpath: str) -> Dict[str, str]:
    """
    Computes global features files paths

    :param global_features: global features
    :param kapture_dirpath: top kapture directory path
    :return: global features to global features file dictionary
    """
    return features_to_filepaths(global_features, kapture_dirpath)


def global_features_check_dir(global_features: kapture.GlobalFeatures, kapture_dirpath: str) -> bool:
    """
    Checks that all keypoints file exist.

    :param global_features: global features
    :param kapture_dirpath: top kapture directory path
    :return: True if they all exist, false otherwise.
    """
    return features_check_dir(global_features, kapture_dirpath)


# matches ##############################################################################################################
def image_matches_from_file(filepath: str) -> np.array:
    """
    Read the image matches
     [keypoint_id_image1, keypoint_id_image2, score]

    :param filepath: path to the file
    :return: matches
    """
    return array_from_file(filepath, dtype=np.float64, dsize=3)


def image_matches_to_file(filepath: str, image_matches: np.array) -> None:
    """
    Writes the image matches to file

    :param filepath: file path
    :param image_matches: image matches
    """

    assert image_matches.dtype == np.float64
    assert image_matches.shape[1] == 3
    array_to_file(filepath, image_matches)


def get_matches_fullpath(image_filename_pair: Optional[Tuple[str, str]] = None, kapture_dirpath: str = '') -> str:
    """
    Computes the full path of the matches file between two images

    :param kapture_dirpath: top kapture directory path
    :param image_filename_pair: image file names
    :return: full path of the matches file
    """

    if image_filename_pair is not None:
        filename = path.join(image_filename_pair[0] + FEATURE_PAIR_PATH_SEPARATOR[kapture.Matches],
                             image_filename_pair[1])
    else:
        filename = None
    return get_features_fullpath(kapture.Matches, kapture_dirpath, filename)


def matches_to_filepaths(matches: kapture.Matches, kapture_dirpath: str = '') -> Dict[Tuple[str, str], str]:
    """
    Computes matches files paths

    :param matches: matches
    :param kapture_dirpath: top kapture directory path
    :return: file to file matches to matches file dictionary
    """
    return {
        image_filename_pair: get_matches_fullpath(image_filename_pair, kapture_dirpath)
        for image_filename_pair in matches
    }


def matching_pairs_from_dirpath(kapture_dirpath: str) -> Iterable[Tuple[str, str]]:
    """
    Read and build Matches from kapture directory tree.
    """
    matches_dirpath = get_matches_fullpath(None, kapture_dirpath)
    # list all files there is
    # filter only match files (the ones endings with .matches)
    matches_filenames = populate_files_in_dirpath(matches_dirpath, FEATURE_FILE_EXTENSION[kapture.Matches])

    # remove the extensions and cut
    matching_pairs = (
        path_secure(matches_filename)[:-len(FEATURE_FILE_EXTENSION[kapture.Matches])
                                      ].split(FEATURE_PAIR_PATH_SEPARATOR[kapture.Matches] + '/')
        for matches_filename in matches_filenames
    )
    matching_pairs = ((matches[0], matches[1]) for matches in matching_pairs if len(matches) == 2)
    return matching_pairs


def matches_check_dir(matches: kapture.Matches, kapture_dirpath: str) -> bool:
    """
    Checks that all matches file exist.

    :param matches: matches
    :param kapture_dirpath: top kapture directory path
    :return: True if they all exist, false otherwise.
    """
    file_list = (get_matches_fullpath(pair, kapture_dirpath) for pair in matches)
    all_files_exists = all(path.exists(filepath) for filepath in file_list)
    return all_files_exists
