# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This files contains IO operations on binary file.
"""
import os
import os.path as path
import numpy as np
import shutil
from typing import Type, Iterable
from tqdm import tqdm
import logging
from kapture.utils.logging import getLogger
from kapture.utils.paths import path_secure
from enum import auto
from kapture.utils.Collections import AutoEnum

logger = getLogger()


# Binary data File IO ##################################################################################################
def array_from_file(filepath: str, dtype: Type, dsize: int) -> np.array:
    """
    Reads a binary file of given type(dtype) and columns (dsize) into a numpy array.

    :param filepath: input path to binary file
    :param dtype: data type (eg. float)
    :param dsize: number of data (of type) per feature (eg. 2 for a simple keypoint)
    :return: numpy array
    """
    if not isinstance(dtype, type):
        raise TypeError('expect type as dtype.')
    if not isinstance(dsize, int) or dsize <= 0:
        raise TypeError('expect positive int as dsize.')

    with open(filepath, 'rb') as file:
        data_array = np.fromfile(file, dtype=dtype)
    data_array = data_array.reshape((-1, dsize))
    return data_array


def array_to_file(filepath: str, data_array: np.array) -> None:
    """
    Writes the numpy array into a binary file.

    :param filepath:
    :param data_array:
    """
    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        data_array.tofile(f, sep='')


########################################################################################################################
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
        # make sure we deal absolute full path
        src = path_secure(path.abspath(src))
        dst = path_secure(path.abspath(dst))
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
