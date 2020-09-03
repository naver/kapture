#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
# This script imports data from a EXIF


import argparse
import logging
import os.path as path
import path_to_kapture  # noqa: F401
import kapture
from kapture.converter.exif.import_exif import import_gps_from_exif

logger = logging.getLogger('exif')


def extract_exif_command_line() -> None:
    """
    Extract GPS coordinates from image EXIF metadata.
    """
    parser = argparse.ArgumentParser(description='Extract GPS coordinates from image EXIF metadata '
                                                 'and store in kapture files.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', '--kapture', required=True,
                        help='input path to kapture data root directory')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    args.input = path.abspath(args.input)
    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v) for k, v in vars(args).items()]))
    # do the job
    import_gps_from_exif(kapture_dirpath=args.input)


if __name__ == '__main__':
    extract_exif_command_line()
