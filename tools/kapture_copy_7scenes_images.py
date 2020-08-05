#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to copy the 7scenes images into the set of 7-scenes kaptures without images.

This is useful to run if you downloaded the 7scenes kapture dataset without the images.
The images are copied from the original dataset that should be first downloaded from its original location:
    https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

The 7 datasets should be downloaded and stored side by side in the same directory.
"""

import argparse
import logging
import os
import os.path as path
import re
import shutil
from tqdm import tqdm

# kapture
import path_to_kapture
import kapture
from kapture.io.records import get_image_fullpath

logger = logging.getLogger('7scenes')

SCENE_SEQUENCE_USAGE = {
    'chess/seq-01': 'mapping',
    'chess/seq-02': 'mapping',
    'chess/seq-03': 'query',
    'chess/seq-04': 'mapping',
    'chess/seq-05': 'query',
    'chess/seq-06': 'mapping',
    'fire/seq-01': 'mapping',
    'fire/seq-02': 'mapping',
    'fire/seq-03': 'query',
    'fire/seq-04': 'query',
    'heads/seq-01': 'query',
    'heads/seq-02': 'mapping',
    'office/seq-01': 'mapping',
    'office/seq-02': 'query',
    'office/seq-03': 'mapping',
    'office/seq-04': 'mapping',
    'office/seq-05': 'mapping',
    'office/seq-06': 'query',
    'office/seq-07': 'query',
    'office/seq-08': 'mapping',
    'office/seq-09': 'query',
    'office/seq-10': 'mapping',
    'pumpkin/seq-01': 'query',
    'pumpkin/seq-02': 'mapping',
    'pumpkin/seq-03': 'mapping',
    'pumpkin/seq-06': 'mapping',
    'pumpkin/seq-07': 'query',
    'pumpkin/seq-08': 'mapping',
    'redkitchen/seq-01': 'mapping',
    'redkitchen/seq-02': 'mapping',
    'redkitchen/seq-03': 'query',
    'redkitchen/seq-04': 'query',
    'redkitchen/seq-05': 'mapping',
    'redkitchen/seq-06': 'query',
    'redkitchen/seq-07': 'mapping',
    'redkitchen/seq-08': 'mapping',
    'redkitchen/seq-11': 'mapping',
    'redkitchen/seq-12': 'query',
    'redkitchen/seq-13': 'mapping',
    'redkitchen/seq-14': 'query',
    'stairs/seq-01': 'query',
    'stairs/seq-02': 'mapping',
    'stairs/seq-03': 'mapping',
    'stairs/seq-04': 'query',
    'stairs/seq-05': 'mapping',
    'stairs/seq-06': 'mapping'
}


def copy_scene_sequence_images(sequence_path: str,
                               kapture_path: str) -> None:
    """
    Copies all color images from the sequence to the corresponding kapture image directory.

    :param sequence_path: the images sequence path
    :param kapture_path: path to the kapture directory
    """
    # Source and destination directories must exist
    if not path.isdir(sequence_path):
        raise ValueError(f"Missing sequence directory {sequence_path}")
    kapture_images_path = get_image_fullpath(kapture_path)
    if not path.exists(kapture_images_path):
        os.mkdir(kapture_images_path)
    if not path.isdir(kapture_images_path):
        raise ValueError(f"Missing kapture images directory {kapture_images_path}")
    # Copy only color image files
    logger.info(f'copying color image files from {sequence_path} to {kapture_images_path}')
    d7s_color_image_re = re.compile(r'frame-(?P<timestamp>\d{6})\.color\.png')
    images_filenames = (fn
                        for dp, _, fs in os.walk(sequence_path) for fn in fs)
    images_files_paths = sorted(list(path.join(sequence_path, filename)
                                for filename in images_filenames
                                if d7s_color_image_re.match(filename)))
    progress_bar = tqdm(total=len(images_files_paths)) if logger.getEffectiveLevel() <= logging.INFO else None
    for image_file_path in images_files_paths:
        shutil.copy2(image_file_path, kapture_images_path)
        progress_bar and progress_bar.update(1)


def copy_7scenes_images(top_7scenes_path: str,
                        top_kapture_path: str) -> None:
    """
    Copies 7-Scenes dataset images into their kapture.

    :param top_7scenes_path: path to the 7scenes top directory path
    :param top_kapture_path: path to kapture top directory
    """
    # First check that all directories are there
    scenes_sequences = list(SCENE_SEQUENCE_USAGE.keys())
    scenes_names = sorted(list(set(path.split(scene_sequence)[0]
                                   for scene_sequence in scenes_sequences)))
    for scene_name in scenes_names:
        scene_path = path.join(top_7scenes_path, scene_name)
        if not path.isdir(scene_path):
            raise ValueError(f"Missing scene '{scene_name}' in {top_7scenes_path}")
        kapture_scene_path = path.join(top_kapture_path, scene_name)
        if not path.isdir(kapture_scene_path):
            raise ValueError(f"Missing kapture scene '{scene_name}' in {top_kapture_path}")

    # Copy sequence by sequence
    for scene_sequence in scenes_sequences:
        sequence_path = path.join(top_7scenes_path, scene_sequence)
        scene_name, sequence_name = path.split(scene_sequence)
        kapture_path = path.join(top_kapture_path, scene_name,
                                 SCENE_SEQUENCE_USAGE[scene_sequence], sequence_name)
        copy_scene_sequence_images(sequence_path, kapture_path)


def copy_7scenes_images_command_line() -> None:
    """
    Copy 7-Scenes dataset color images in the 7scenes kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='Copies 7-Scenes Dataset RGB images files to the kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.INFO, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    # import ###########################################################################################################
    parser.add_argument('-d', '--dataset', required=True, help='7-Scenes original dataset top directory path '
                                                               '(with 7 sub-directories for chess, fire, ...)')
    parser.add_argument('-k', '--kapture', required=True, help='7-scenes kapture top directory with 7 sub-directories.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)

    copy_7scenes_images(args.dataset, args.kapture)


if __name__ == '__main__':
    copy_7scenes_images_command_line()
