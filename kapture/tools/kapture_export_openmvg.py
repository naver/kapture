#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export a kapture into an openmvg file.
"""

import logging
import os.path as path
from typing import Optional
# kapture
from kapture.io.records import TransferAction
import kapture.converter.openmvg.export_openmvg as openmvg_export
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('openmvg')


def export_openmvg(
        input: str,
        output: Optional[str] = None,
        sfm_data: Optional[str] = None,
        images: Optional[str] = None,
        regions: Optional[str] = None,
        matches: Optional[str] = None,
        image_transfer: TransferAction = TransferAction.skip,
        image_path_flatten: bool = False,
        keypoints_type: Optional[str] = None,
        descriptors_type: Optional[str] = None,
        use_v2_intrinsics_format: bool = False,
        force: bool = False,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False,
) -> None:
    """
    Exports from kapture format to openMVG JSON file.

    :param input: path to input kapture data root directory
    :param output: root path to openmvg data structure (sfm_data, images, regions and matches sub paths are guessed).
    :param sfm_data: path to output openMVG sfm_data JSON file.
    :param images: path to output openMVG image directory.
    :param regions: path to output openMVG regions directory (for features and descriptors).
    :param matches: path to output openMVG matches file.
    :param image_transfer: what to do with images: root_link: link to the root of the images directory;
                           link_absolute: absolute individual file link; link_relative: relative individual file link;
                           copy: copy file instead of creating link; move: move file instead of creating link;
                           skip: do not create links.
    :param image_path_flatten: flatten image subpath, to make sure there is no collision in image names.
    :param keypoints_type: kapture keypoints type.
    :param descriptors_type: kapture descriptors type.
    :param use_v2_intrinsics_format: do not use value0 for pinhole_brown_t2 and fisheye cameras.
    :param force: delete output file if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    if output is None and sfm_data is None:
        raise ValueError('at least one output path should be given.')

    # auto complete using output
    if output and sfm_data is None:
        sfm_data = path.join(output, 'sfm_data.json')
        logger.debug(f'guessing output sfm_data is {sfm_data}')
    if output and images is None:
        images = path.join(output, 'images')
        logger.debug(f'guessing output images is {images}')
    if output and regions is None:
        regions = path.join(output, 'matches')
        logger.debug(f'guessing output regions is {regions}')
    if output and matches is None:
        matches = path.join(output, 'matches', 'matches.txt')
        logger.debug(f'guessing output matches is {matches}')

    # no image dir == skip transfer
    if images is None:
        logger.info('No images directory provided, skipping image files transfer')
        image_transfer = TransferAction.skip
    if image_transfer == TransferAction.skip:
        images = None

    openmvg_export.export_openmvg(input,
                                  sfm_data,
                                  images,
                                  regions,
                                  matches,
                                  image_transfer,
                                  image_path_flatten,
                                  keypoints_type,
                                  descriptors_type,
                                  use_v2_intrinsics_format,
                                  force
                                  )


def main():
    run_cli(export_openmvg, aliases={
        '--input': ['-k', '--kapture'],
        '--output': ['--openmvg'],
        '--sfm_data': ['-s'],
        '--images': ['-im'],
        '--regions': ['-r'],
        '--matches': ['-m'],
        '--image_transfer': ['--image_action'],
        '--keypoints_type': ['-kpt'],
        '--descriptors_type': ['-desc'],
    })


if __name__ == '__main__':
    main()
