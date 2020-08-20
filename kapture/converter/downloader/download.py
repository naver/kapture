# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import requests
import os
import os.path as path
from tqdm import tqdm
from typing import Optional
import logging

logger = logging.getLogger('downloader')


def get_remote_file_size(url: str):
    """
    return the total file size on the remote url.

    :param url: input full url of the file.
    :return : int of file size in bytes, or None if unknown
    """
    # hack: use requests.get(range0-10) instead of requests.header. Header does not follow redirections in some cases.
    headers = {"Range": "bytes=0-10"}  # only retrieve the first bytes,
    response = requests.get(url, allow_redirects=True, headers=headers)
    # content length is not relevant (always 11, since that is what was requested)
    # lets look to Content-Range, that is formatted as 0-11/total
    if 'Content-Range' not in response.headers or '/' not in response.headers['Content-Range']:
        return None
    nb_bytes_total = int(response.headers['Content-Range'].split('/')[1])
    return nb_bytes_total


def download_file_resume(url: str,
                         filepath: str,
                         resume_byte_pos: Optional[int] = None):
    """
    resume (or start if no pos given) the dataset download.

    :param url: input full url of the file to be downloaded.
    :param filepath: input full path where to save the file.
    :param resume_byte_pos: input position in bytes where to resume the Download
    """

    file_size_online = get_remote_file_size(url)
    # Append information to resume download at specific byte position to header
    resume_header = ({'Range': f'bytes={resume_byte_pos}-'}
                     if resume_byte_pos else None)
    # Establish connection
    response = requests.get(url, stream=True, headers=resume_header, allow_redirects=True)
    # Set configuration
    block_size = 1024  # 1Ko
    initial_pos = resume_byte_pos if resume_byte_pos else 0
    mode = 'ab' if resume_byte_pos else 'wb'
    hide_progress = logger.getEffectiveLevel() > logging.INFO
    with open(filepath, mode) as f:
        with tqdm(total=file_size_online, unit='B',
                  unit_scale=True, unit_divisor=block_size,
                  desc=filepath, initial=initial_pos,
                  ascii=True, miniters=1,
                  disable=hide_progress) as pbar:
            for chunk in response.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(url, filepath):
    """
     Starts or resumes the download if already started.

    :param url: input full url of the file to be downloaded.
    :param filepath: input full path where to save the file.
     """
    resume_position = None
    if path.isfile(filepath):
        logger.debug('file is already (partially) there.')
        # file already there (at least partially)
        file_size_online = get_remote_file_size(url)
        file_size_local = int(path.getsize(filepath))
        if file_size_online is None:
            raise ValueError('Unable to retrieve file size on remote url.')

        if file_size_online == file_size_local:
            logger.debug(f'file {filepath} already downloaded.')
            return

        logger.debug(f'resume download from {file_size_local / file_size_online * 100.:4.1f}%')
        resume_position = file_size_local

    logger.debug(f'start downloading "{filepath}"\n\tfrom: "{url}".')
    os.makedirs(path.dirname(filepath), exist_ok=True)
    download_file_resume(url, filepath, resume_position)
