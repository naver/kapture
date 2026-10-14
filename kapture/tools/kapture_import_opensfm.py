#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import logging
import os.path as path
from kapture.io.records import TransferAction
import kapture.converter.opensfm.import_opensfm as opensfm_import
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


logger = logging.getLogger('opensfm')


def import_opensfm(
        input: str,
        output: str,
        image_transfer: TransferAction = TransferAction.link_absolute,
        force: bool = False,
        keypoints_type: str = 'HessianAffine',
        descriptors_type: str = 'HOG',
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False
) -> None:
    """
    Imports openSfM to kapture.

    :param input: path to OpenSfM root directory.
    :param output: output directory where to save kapture files.
    :param image_transfer: how to import images
    :param force: delete kapture if already exists.
    :param keypoints_type: kapture keypoints type.
    :param descriptors_type: kapture descriptors type.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    input = path.normpath(path.abspath(input))
    output = path.normpath(path.abspath(output))
    opensfm_import.import_opensfm(input, output, force, image_transfer, keypoints_type, descriptors_type)


def main():
    run_cli(import_opensfm, aliases={
        '--input': ['--opensfm'],
        '--output': ['-k', '--kapture'],
        '--image_transfer': ['--transfer'],
    })


if __name__ == '__main__':
    main()
