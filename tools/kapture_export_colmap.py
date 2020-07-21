#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script exports kapture data to a COLMAP database and/or reconstruction files
"""

import argparse
import logging
import sys
import path_to_kapture  # enables import kapture
import kapture
import kapture.utils.logging
from kapture.converter.colmap.export_colmap import export_colmap


logger = logging.getLogger('colmap')


def colmap_command_line():
    """
    Do the export of kapture data using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='exports kapture data to Colmap database and/or text files.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete colmap if already exists.')
    # create the parser for the export command #########################################################################
    parser.add_argument('-i', '-k', '--kapture', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-db', '--database', required=True,
                        help='database output path.')
    parser.add_argument('-txt', '--reconstruction',
                        help='text reconstruction output path.')
    parser.add_argument('-rig', '--rig',
                        help='json rig output path.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # for debug, let kapture express itself.
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(f'{sys.argv[0]} \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()))

    logger.info('exporting colmap ...')
    export_colmap(args.kapture, args.database, args.reconstruction, args.rig, args.force)
    logger.info('done.')


if __name__ == '__main__':
    colmap_command_line()
