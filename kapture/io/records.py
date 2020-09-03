# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This files contains IO operations on Record related data.
"""

import os
import os.path as path
from typing import Dict, Optional, Union, Iterable, Tuple
import numpy as np
import kapture
from kapture.utils.paths import path_secure
from kapture.utils.logging import getLogger
from kapture.io.binary import TransferAction, transfer_files_from_dir, array_from_file, array_to_file

logger = getLogger()

# Records files related functions ######################################################################################
RECORD_DATA_DIRNAME = path_secure(path.join('sensors', 'records_data'))


########################################################################################################################
def get_record_fullpath(
        kapture_dirpath: str = '',
        record_filename: Optional[str] = None) -> str:
    """
    Returns full path to subdirectory containing the binary files of the record type.
            Optionally, can give a the file name.
    :param kapture_dirpath: input path to kapture directory.
    :param record_filename: optional input record filename (eg. image filename).
    :return: the record full path
    """
    feature_filename = record_filename or ''
    return path_secure(path.join(kapture_dirpath, RECORD_DATA_DIRNAME, feature_filename))


def records_to_filepaths(
        records: Union[kapture.RecordsCamera,
                       kapture.RecordsDepth,
                       kapture.RecordsWifi,
                       kapture.RecordsLidar,
                       kapture.RecordsGnss],
        kapture_dirpath: str
) -> Dict[str, str]:
    """
    Computes filepaths for records.

    :param records: records
    :param kapture_dirpath: top kapture directory path
    :return: records name to records file path dictionary
    """
    return {filename: path_secure(path.join(kapture_dirpath, RECORD_DATA_DIRNAME, filename))
            for _, _, filename in kapture.flatten(records)}


def guess_filepaths_from_filenames(
        dirpath: str,
        filenames: Iterable[str]
):
    """ returns a generator that prepend the directory path to the given filenames."""
    return (path_secure(path.join(dirpath, record_filename))
            for record_filename in filenames)


def import_record_data_from_dir_link_dir(
        source_record_dirpath: str,
        kapture_dirpath: str,
        do_relative_link: bool = False
) -> None:
    """
    Imports record_data from a given directory:
    This do not actually copy any files, but make a symbolic link on the root path instead.

    :param source_record_dirpath: input path to directory where to import data from.
    :param kapture_dirpath: input root path to kapture root directory, where to import data to.
    :param do_relative_link: if True, do relative link else absolute link.
    """
    assert path.isdir(source_record_dirpath)
    kapture_record_dirpath = get_record_fullpath(kapture_dirpath)
    if do_relative_link:
        source_record_dirpath = path.relpath(source_record_dirpath, kapture_record_dirpath)
    try:  # on windows, symlink requires some privileges, and may crash if not
        os.symlink(source_record_dirpath, kapture_record_dirpath)
    except OSError as e:
        logger.critical('unable to create symlink on image directory, due to privilege restrictions.')
        raise e


def import_record_data_from_dir_auto(
        source_record_dirpath: str,
        destination_kapture_dirpath: str,
        filename_list: Iterable[str],
        copy_strategy: TransferAction = TransferAction.copy
) -> None:
    """
    Imports record_data from a given directory.
    Automatically replicate the folder hierarchy from source to kapture record directory.
    If you want to change the file organisation use import_record_data_from_dir_explicit instead.
    The actual import can be done using actual file copy (copy/move), or symlinks (absolute/relative).

    :param source_record_dirpath: input path to directory where to import data from.
    :param destination_kapture_dirpath: input root path to kapture root directory, where to import data to.
    :param filename_list: input list of filenames to import (filenames are relative path to source_record_dirpath).
                    This image list can be obtained from file walking (populate_files_in_dirpath)
                    or using an already populated kapture.RecordsData. Prefer RecordsData since it will only copy
                    required files, eg:
                    filename_list = [f for _, _, f in kapture.flatten(kapture_data.records_camera)]
    :param copy_strategy:
    """
    if copy_strategy == TransferAction.root_link:
        import_record_data_from_dir_link_dir(source_record_dirpath, destination_kapture_dirpath)
    else:
        source_filepath_list = (path.join(source_record_dirpath, record_filename)
                                for record_filename in filename_list)
        kapture_filepath_list = (get_record_fullpath(destination_kapture_dirpath, record_filename)
                                 for record_filename in filename_list)
        transfer_files_from_dir(
            source_filepath_list,
            kapture_filepath_list,
            copy_strategy
        )


# images ###############################################################################################################
def get_image_fullpath(kapture_dir_path: str, image_filename: Optional[str] = None) -> str:
    """
    Get the full path of the image file in the kapture.
     If image_filename is missing, this gives the top directory (records_data).

    :param kapture_dir_path: the kapture top directory
    :param image_filename: optional image file name
    :return: Image file full path
    """
    return get_record_fullpath(kapture_dir_path, image_filename)


def images_to_filepaths(images: kapture.RecordsCamera, kapture_dirpath: str) -> Dict[str, str]:
    """
    Computes filepaths for image records.

    :param images: images records
    :param kapture_dirpath: top kapture directory path
    :return: images name to images file path dictionary
    """
    return records_to_filepaths(images, kapture_dirpath)


# depth maps file paths ################################################################################################
def get_depth_map_fullpath(kapture_dir_path: str, depth_map_filename: Optional[str] = None) -> str:
    """
    Get the full path of the depth map file in the kapture.
     If depth_map_filename is missing, this gives the top directory (records_data).

    :param kapture_dir_path: the kapture top directory
    :param depth_map_filename: optional image file name
    :return: Depth map file full path
    """
    return get_record_fullpath(kapture_dir_path, depth_map_filename)


def depth_maps_to_filepaths(depth_records: kapture.RecordsDepth, kapture_dirpath: str) -> Dict[str, str]:
    """
    Computes filepaths for depth maps records.

    :param images: images records
    :param kapture_dirpath: top kapture directory path
    :return: images name to images file path dictionary
    """
    return records_to_filepaths(depth_records, kapture_dirpath)


# depth maps IO ########################################################################################################
def records_depth_from_file(filepath: str, size: Tuple[int, int]) -> np.array:
    """
    Load the depth map from binary file to a numpy array.

    :param filepath: path to the file
    :param size: [width, height]
    :return: the depth map as a numpy array
    """
    assert isinstance(size, tuple) and len(size) == 2
    dtype = kapture.RecordsDepth.dtype
    dsize = int(size[0] * size[1])
    bitmap = array_from_file(filepath, dtype, dsize)
    bitmap = bitmap.reshape((size[1], size[0]))
    return bitmap


def records_depth_to_file(filepath: str, depth_map: np.array) -> None:
    """
    Writes the depth map from numpy array to binary file.

    :param filepath: file path
    :param depth_map: depth map as a numpy array
    """
    array_to_file(filepath, depth_map)
