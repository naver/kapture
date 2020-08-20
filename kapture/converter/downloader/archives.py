# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import tarfile
import hashlib
import logging

logger = logging.getLogger('downloader')


def untar_file(archive_filepath: str,
               install_dirpath: str):
    """
    Equivalent to tar -xf <archive_filepath> -C <install_dirpath>

    :param archive_filepath: input full path to the archive file.
    :param install_dirpath: input full path to directory where to extract.
    """
    logger.debug(f'extracting\n\tfrom: {archive_filepath}\n\tto  : {install_dirpath}')
    # make sure directory exists
    os.makedirs(install_dirpath, exist_ok=True)
    with tarfile.open(archive_filepath, 'r:*') as archive:
        archive.extractall(install_dirpath)


def compute_sha256sum(archive_filepath: str):
    """
    Computes the sha256sum on the given file.

    :param archive_filepath: input full path to file.
    :return the sha256sum.
    """
    sha256_hash = hashlib.sha256()
    with open(archive_filepath, 'rb') as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
