#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script exports kapture data to a COLMAP database and/or reconstruction files
"""

import logging
from typing import Optional
import kapture.converter.colmap.export_colmap as colmap_export
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


logger = logging.getLogger('colmap')


def export_colmap(
        input: str,
        database: str,
        reconstruction: Optional[str] = None,
        rig: Optional[str] = None,
        keypoints_type: Optional[str] = None,
        descriptors_type: Optional[str] = None,
        pairsfile_path: Optional[str] = None,
        force: bool = False,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False,
) -> None:
    """
    Exports kapture data to Colmap database and/or text files.

    :param input: input path to kapture data root directory
    :param database: database output path.
    :param reconstruction: text reconstruction output path.
    :param rig: json rig output path.
    :param keypoints_type: kapture keypoints type.
    :param descriptors_type: kapture descriptors type.
    :param pairsfile_path: text file in the csv format; where each line is image_name1, image_name2, score
                           which contains the matches to export
    :param force: delete colmap if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    logger.info('exporting colmap ...')
    colmap_export.export_colmap(input, database, reconstruction,
                                keypoints_type, descriptors_type,
                                rig, force, pairsfile_path)
    logger.info('done.')


def main():
    run_cli(export_colmap, aliases={
        '--input': ['-k', '--kapture'],
        '--database': ['-db'],
        '--reconstruction': ['-txt'],
        '--rig': ['-rig'],
        '--keypoints_type': ['-kpt'],
        '--descriptors_type': ['-desc'],
    })


if __name__ == '__main__':
    main()
