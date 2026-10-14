#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export a kapture to openSfM.
"""

import logging
import os.path as path
from typing import Optional
from kapture.io.records import TransferAction
import kapture.converter.opensfm.export_opensfm as opensfm_export
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


logger = logging.getLogger('opensfm')


def export_opensfm(
        input: str,
        output: str,
        image_transfer: TransferAction = TransferAction.link_absolute,
        force: bool = False,
        keypoints_type: Optional[str] = None,
        descriptors_type: Optional[str] = None,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False,
) -> None:
    """
    Exports from kapture to openSfM.

    :param input: path to kapture root directory.
    :param output: directory where to save OpenSfM files.
    :param image_transfer: how to export images
    :param force: delete opensfm data if already exists.
    :param keypoints_type: kapture keypoints type.
    :param descriptors_type: kapture descriptors type.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    input = path.normpath(path.abspath(input))
    output = path.normpath(path.abspath(output))
    opensfm_export.export_opensfm(input, output, force, image_transfer, keypoints_type, descriptors_type)


def main():
    run_cli(export_opensfm, aliases={
        '--input': ['-k', '--kapture'],
        '--output': ['--opensfm'],
        '--image_transfer': ['--transfer'],
        '--keypoints_type': ['-kpt'],
        '--descriptors_type': ['-desc'],
    })


if __name__ == '__main__':
    main()
