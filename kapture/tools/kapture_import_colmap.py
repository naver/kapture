#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports data from a COLMAP database and/or reconstruction files

It will assume that if all rotation parameters are null, there is no prior rotation, and if all translation parameters
are null, there is no prior translation (using null values instead of zeros)
"""

import logging
import os
from typing import Optional

import kapture
import kapture.io.csv
import kapture.io.structure
from kapture.io.records import TransferAction
import kapture.converter.colmap.import_colmap as colmap_import
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('colmap')


def import_colmap(
        output: str,
        database: Optional[str] = None,
        reconstruction: Optional[str] = None,
        images: Optional[str] = None,
        image_transfer: TransferAction = TransferAction.link_absolute,
        rig: Optional[str] = None,
        keypoints_type: str = 'SIFT',
        descriptors_type: str = 'SIFT',
        skip_reconstruction: bool = False,
        geometric_filtering: bool = True,
        force: bool = False,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False,
) -> None:
    """
    Imports data from colmap database and/or text files.

    :param output: output directory where to save kapture files.
    :param database: path to COLMAP database file.
    :param reconstruction: path to COLMAP reconstruction triplet text file.
    :param images: path to images directory.
    :param image_transfer: how to import images
    :param rig: path to COLMAP rig json file.
    :param keypoints_type: kapture keypoints type.
    :param descriptors_type: kapture descriptors type.
    :param skip_reconstruction: skip the import of the kapture/reconstruction,
                                ie. keypoints, descriptors, matches, points3d, observations.
    :param geometric_filtering: remove matches failing geometric verification.
    :param force: delete kapture if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    if image_transfer == TransferAction.root_link:
        logger.fatal('Root linking the images is not supported yet.')
        return
    logger.debug('cleaning output directory')
    os.makedirs(output, exist_ok=True)
    kapture.io.structure.delete_existing_kapture_files(output, force_erase=force)

    logger.info('importing colmap ...')
    kapture_data = colmap_import.import_colmap(output,
                                               database,
                                               reconstruction,
                                               images,
                                               rig,
                                               keypoints_type,
                                               descriptors_type,
                                               not geometric_filtering,
                                               skip_reconstruction,
                                               force,
                                               image_transfer if images is not None else TransferAction.skip
                                               )

    logger.info('saving to kapture  ...')
    logger.debug(f'\t"{output}"')
    kapture.io.csv.kapture_to_dir(output, kapture_data)


def main():
    run_cli(import_colmap, aliases={
        '--output': ['-k', '--kapture'],
        '--database': ['-db'],
        '--reconstruction': ['-txt'],
        '--images': ['-im'],
        '--rig': ['-rig'],
        '--keypoints_type': ['-kpt'],
        '--descriptors_type': ['-desc'],
    })


if __name__ == '__main__':
    main()
