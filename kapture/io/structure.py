# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Internal structure related operations.
"""

import os
import os.path as path
from shutil import rmtree
from typing import List, Optional
from .features import FEATURES_DATA_DIRNAMES
from .records import get_record_fullpath
from .csv import CSV_FILENAMES
from kapture.core.Records import RecordsFilePath
from kapture.utils.logging import getLogger
from kapture.utils.paths import path_secure


def delete_existing_kapture_files(
        dirpath: str,
        force_erase: bool,
        only: Optional[List[type]] = None,
        skip: Optional[List[type]] = None):
    """
    Deletes all existing files / directories at dirpath that corresponds to kapture data.
    do not use only and skip at the same time.

    :param dirpath:
    :param force_erase: do not ask user confirmation.
    :param only: can be used to select files / directories to be removed.
    :param skip: can be used to select files / directories to be kept.
    :return:
    """
    assert only is None or isinstance(only, list)
    assert skip is None or isinstance(skip, list)

    to_keep_csv_list = []
    to_keep_features_list = []
    if skip:
        to_keep_csv_list = [dtype for dtype in skip if dtype not in FEATURES_DATA_DIRNAMES.keys()]
        to_keep_features_list = [dtype for dtype in skip if dtype not in CSV_FILENAMES.keys()]
    if only:
        to_keep_csv_list = [dtype for dtype in CSV_FILENAMES.keys() if dtype not in only]
        to_keep_features_list = [dtype for dtype in FEATURES_DATA_DIRNAMES.keys() if dtype not in only]
    # Compute if we must keep records whose values are stored in files
    to_keep_records_with_file = (issubclass(dtype, RecordsFilePath) for dtype in to_keep_csv_list+to_keep_features_list)
    must_keep_records_dir = any(to_keep_records_with_file)
    dirpath = path_secure(dirpath)
    csv_filepaths = [
        path.join(dirpath, filename)
        for dtype, filename in CSV_FILENAMES.items()
        if dtype not in to_keep_csv_list]
    features_dirpaths = [
        path.join(dirpath, dirname)
        for dtype, dirname in FEATURES_DATA_DIRNAMES.items()
        if dtype not in to_keep_features_list]
    records_dirpath = get_record_fullpath(dirpath)
    # remove files (start with deepest/longest paths to avoid deleting dirs before files).
    existing_paths = list(reversed(sorted({pathval
                                           for pathval in csv_filepaths + features_dirpaths + [records_dirpath]
                                           if path.lexists(pathval)})))
    if existing_paths and must_keep_records_dir:
        existing_paths.remove(records_dirpath)
    # if any
    if existing_paths:
        existing_paths_as_string = ', '.join(f'"{path.relpath(p, dirpath)}"' for p in existing_paths)
        # ask for permission
        to_delete = (force_erase or
                     (input(f'{existing_paths_as_string} already in "{dirpath}".'
                            ' Delete ? [y/N]').lower() == 'y'))

        # delete all or quit
        if to_delete:
            getLogger().info('deleting already'
                             f' existing {existing_paths_as_string}')
            for pathval in existing_paths:
                if path.islink(pathval) or path.isfile(pathval):
                    os.remove(pathval)
                else:
                    rmtree(pathval)
        else:
            raise ValueError(f'{existing_paths_as_string} already exist')
