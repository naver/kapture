import requests
import os
import os.path as path
from tqdm import tqdm
from typing import Optional
import logging

logger = logging.getLogger('downloader')


def download_file_resume(url: str,
                         filepath: str,
                         resume_byte_pos: Optional[int] = None):
    """
    resume (or start if no pos given) the dataset download.

    :param url: input full url of the file to be downloaded.
    :param filepath: input full path where to save the file.
    :param resume_byte_pos: input position in bytes where to resume the Download
    """

    response = requests.head(url)
    file_size_online = int(response.headers.get('content-length', 0))
    # Append information to resume download at specific byte position to header
    resume_header = ({'Range': f'bytes={resume_byte_pos}-'}
                     if resume_byte_pos else None)
    # Establish connection
    response = requests.get(url, stream=True, headers=resume_header)
    # Set configuration
    block_size = 1024  # 1Ko
    initial_pos = resume_byte_pos if resume_byte_pos else 0
    mode = 'ab' if resume_byte_pos else 'wb'
    with open(filepath, mode) as f:
        with tqdm(total=file_size_online, unit='B',
                  unit_scale=True, unit_divisor=block_size,
                  desc=filepath, initial=initial_pos,
                  ascii=True, miniters=1,
                  disable=logger.getEffectiveLevel() >= logging.CRITICAL) as pbar:
            for chunk in response.iter_content(32 * block_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(url, filepath):
    """
     Starts or resumes the download if already started.

    :param url: input full url of the file to be downloaded.
    :param filepath: input full path where to save the file.
     """
    logger.debug(f'downloading {filepath}')
    response = requests.head(url)
    if path.isfile(filepath):
        logger.debug('file is already (partially) there.')
        # file already there (at least partially)
        file_size_online = int(response.headers.get('content-length', 0))
        file_size_local = int(path.getsize(filepath))
        if file_size_online == file_size_local:
            logger.info(f'file {filepath} already downloaded.')
        else:
            logger.info(f'resume download from {file_size_local / file_size_online * 100.:4.1f}%')
            download_file_resume(url, filepath, file_size_local)
    else:
        logger.info(f'start downloading {filepath}.')
        os.makedirs(path.dirname(filepath), exist_ok=True)
        download_file_resume(url, filepath)
