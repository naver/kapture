import os
import os.path as path
import tarfile
from typing import Optional
import logging

logger = logging.getLogger('downloader')


def untar_file(archive_filepath,
                       install_dirpath):
    logger.debug(f'extracting\n\tfrom: {archive_filepath}\n\tto  : {install_dirpath}')
    # make sure directory exists
    os.makedirs(install_dirpath, exist_ok=True)
    with tarfile.open(archive_filepath, 'r:*') as archive:
        archive.extractall(install_dirpath)
    # cleaning tar
    logger.debug(f'cleaning {archive_filepath}')
    os.remove(archive_filepath)
