# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This files contains IO operations on Record related data.
"""

from enum import auto
import logging
import os
import os.path as path
import shutil
from typing import Dict, Optional, Union, Iterable
from tqdm import tqdm

import kapture
from kapture.utils.paths import path_secure
from kapture.utils.Collections import AutoEnum
from kapture.utils.logging import getLogger

logger = getLogger()

# Records files related functions ######################################################################################
RECORD_DATA_DIRNAME = path_secure(path.join('sensors', 'records_data'))


class TransferAction(AutoEnum):
    """
    All possible operations when transferring recorded data (eg. images, Wifi, IMU, ...)
    or computed data (features, keypoints, ...).
    Transferring means importing, exporting, and more generally going from one representation
    to another representation of the data.
    """
    skip = auto()
    root_link = auto()
    copy = auto()
    move = auto()
    link_absolute = auto()
    link_relative = auto()


########################################################################################################################
# transfer files functions
def transfer_files_from_dir_link(
        source_filepath_list: Iterable[str],
        destination_filepath_list: Iterable[str],
        force_overwrite: bool = False,
        do_relative_link: bool = False
) -> None:
    """
    Transfer every files by linking given from the source list to destination list.
    The matching between source and kapture files are explicitly given.

    :param source_filepath_list: input list of source files. Uses guess_filepaths to obtains it from filenames.
    :param destination_filepath_list: input list of destination files (in kapture tree).
    :param force_overwrite: if True, overwrite destination file.
    :param do_relative_link: if True, do relative links else absolute links.
    """
    hide_progress_bar = logger.getEffectiveLevel() > logging.INFO
    for src, dst in tqdm(zip(source_filepath_list, destination_filepath_list), disable=hide_progress_bar):
        os.makedirs(path.dirname(dst), exist_ok=True)
        if force_overwrite and path.lexists(dst):
            os.remove(dst)
        try:  # on windows, symlink requires some privileges, and may crash if not
            if do_relative_link:
                src = path.relpath(src, path.dirname(dst))
            os.symlink(src, dst)
        except OSError as e:
            logger.critical('unable to create symlink on image directory, due to privilege restrictions.')
            raise e


def transfer_files_from_dir_copy(
        source_filepath_list: Iterable[str],
        destination_filepath_list: Iterable[str],
        force_overwrite: bool = False,
        delete_source: bool = False
) -> None:
    """
    Transfer every files by copying given from the source list to destination list.
    The matching between source and kapture files are explicitly given.
    If delete_source is activated, it moves files instead copying them.

    :param source_filepath_list: input list of absolute path to source files.
    :param destination_filepath_list: input list of absolute path to destination files.
    :param force_overwrite: if True, overwrite destination file.
    :param delete_source: if True, delete the imported files from source_record_dirpath.
    """
    hide_progress_bar = logger.getEffectiveLevel() > logging.INFO
    for src, dst in tqdm(zip(source_filepath_list, destination_filepath_list), disable=hide_progress_bar):
        os.makedirs(path.dirname(dst), exist_ok=True)
        if force_overwrite and path.lexists(dst):
            os.remove(dst)
        if delete_source:
            shutil.move(src, dst)
        else:
            shutil.copyfile(src, dst)


def transfer_files_from_dir(
        source_filepath_list: Iterable[str],
        destination_filepath_list: Iterable[str],
        copy_strategy: TransferAction = TransferAction.copy,
        force_overwrite: bool = False
) -> None:
    """
    Transfer files (copy or link files) from source to destination.
    The matching between source and kapture files are explicitly given.
    The actual import can be done using actual file copy (copy/move), or symlinks (absolute/relative).

    :param source_filepath_list: input list of source files.
    :param destination_filepath_list: input list of destination files (in kapture tree).
    :param copy_strategy: transfer strategy to apply.
    :param force_overwrite: if True, overwrite destination file.
    """
    if TransferAction.skip == copy_strategy:
        return

    elif TransferAction.copy == copy_strategy or TransferAction.move == copy_strategy:
        transfer_files_from_dir_copy(
            source_filepath_list=source_filepath_list,
            destination_filepath_list=destination_filepath_list,
            force_overwrite=force_overwrite,
            delete_source=(TransferAction.move == copy_strategy)
        )

    elif TransferAction.link_absolute == copy_strategy or TransferAction.link_relative == copy_strategy:
        transfer_files_from_dir_link(
            source_filepath_list=source_filepath_list,
            destination_filepath_list=destination_filepath_list,
            force_overwrite=force_overwrite,
            do_relative_link=(TransferAction.link_relative == copy_strategy)
        )

    else:
        raise ValueError(f'Unsupported copy action {copy_strategy}')


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
        records: Union[kapture.RecordsCamera, kapture.RecordsWifi, kapture.RecordsLidar, kapture.RecordsGnss],
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
     If image file name is missing, this gives the top directory of the images.

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
