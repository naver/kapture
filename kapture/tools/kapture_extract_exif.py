#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# This script imports data from a EXIF


import logging
import os.path as path
from kapture.converter.exif.import_exif import import_gps_from_exif
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('exif')


def extract_exif(input: str,
                 verbose: Verbosity = DEFAULT_VERBOSITY,
                 quiet: bool = False) -> None:
    """
    Extract GPS coordinates from image EXIF metadata and store in kapture files.

    :param input: path to kapture data root directory
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    input = path.abspath(input)
    import_gps_from_exif(kapture_dirpath=input)


def main():
    run_cli(extract_exif, aliases={'--input': ['--kapture']})


if __name__ == '__main__':
    main()
