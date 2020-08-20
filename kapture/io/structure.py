# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Internal structure related operations.
"""

import os
import os.path as path
from shutil import rmtree
from typing import List, Optional
from .features import FEATURES_DATA_DIRNAMES
from .records import RECORD_DATA_DIRNAME
from .csv import CSV_FILENAMES
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

    dirpath = path_secure(dirpath)
    csv_filepaths = [
        path.join(dirpath, filename)
        for dtype, filename in CSV_FILENAMES.items()
        if (not only and not skip) or (only and dtype in only) or (skip and dtype not in skip)]
    features_dirpaths = [
        path.join(dirpath, dirname)
        for dtype, dirname in FEATURES_DATA_DIRNAMES.items()
        if (not only and not skip) or (only and dtype in only) or (skip and dtype not in skip)]
    records_dirpaths = [RECORD_DATA_DIRNAME]
    # remove existing_files files (start with deepest/longest paths to avoid to delete files before dirs).
    existing_paths = list(reversed(sorted({pathval
                                           for pathval in csv_filepaths + features_dirpaths + records_dirpaths
                                           if path.isfile(pathval) or path.isdir(pathval)})))
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
