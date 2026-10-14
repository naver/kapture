#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports an openmvg file in the kapture format.
"""

import logging
import os.path as path
from typing import Optional

# kapture
from kapture.io.records import TransferAction
# openmvg
import kapture.converter.openmvg.import_openmvg as openmvg_import
from kapture.converter.openmvg.openmvg_commons import OPENMVG_DEFAULT_JSON_FILE_NAME
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('openmvg')


def import_openmvg(
        output: str,
        input: Optional[str] = None,
        sfm_data: Optional[str] = None,
        regions: Optional[str] = None,
        matches: Optional[str] = None,
        image_transfer: TransferAction = TransferAction.root_link,
        force: bool = False,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False
) -> None:
    """
    Imports from openMVG JSON file to Kapture format.

    :param output: top directory where to save kapture files.
    :param input: path to openMVG directory, will automatically look for sfm_data, regions and pair files.
    :param sfm_data: path to openMVG sfm_data data file.
    :param regions: path to openMVG directory containing region files (feat, desc).
    :param matches: path to openMVG matches file (eg. matches.f.txt)
    :param image_transfer: what to do with images (root_link: link to the root of the images directory)
    :param force: silently delete kapture data if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    # argument logic.
    if input:
        input = path.abspath(input)
        # automatically look for sfm_data, regions and pair files if not specified
        if sfm_data is None:
            sfm_data = path.join(input, OPENMVG_DEFAULT_JSON_FILE_NAME)
        if regions is None:
            regions = input
        if matches is None:
            matches = path.join(input, 'matches.f.txt')

    # normalizes paths
    if sfm_data is not None:
        sfm_data = path.normpath(path.abspath(sfm_data))
    if regions is not None:
        regions = path.normpath(path.abspath(regions))
    if matches is not None:
        matches = path.normpath(path.abspath(matches))
        assert matches.endswith('.txt')

    # sanity check
    if all(i is None for i in [sfm_data, regions, matches]):
        raise ValueError('Need openMVG files as input.')

    openmvg_import.import_openmvg(sfm_data,
                                  regions,
                                  matches,
                                  output,
                                  image_transfer,
                                  force)


def main():
    try:
        run_cli(import_openmvg, aliases={
            '--input': ['--openmvg'],
            '--sfm_data': ['-s'],
            '--regions': ['-r'],
            '--matches': ['-m'],
            '--output': ['-k', '--kapture'],
            '--image_transfer': ['--image_action'],
        })
    except (AssertionError, LookupError, OSError):
        logger.error('Fatal error', exc_info=True)


if __name__ == '__main__':
    main()
