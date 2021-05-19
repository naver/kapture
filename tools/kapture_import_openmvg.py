#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports an openmvg file in the kapture format.
"""

import argparse
import logging
import sys
import os.path as path

# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture.utils.logging
from kapture.io.records import TransferAction
# openmvg
from kapture.converter.openmvg.import_openmvg import import_openmvg
from kapture.converter.openmvg.openmvg_commons import OPENMVG_DEFAULT_JSON_FILE_NAME

logger = logging.getLogger('openmvg')


def import_openmvg_command_line() -> None:
    """
    Do the openmvg to kapture import using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='Imports from openMVG JSON file to Kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-y', '--force', action='store_true', default=False,
                        help='silently delete kapture data if already exists.')
    # create the parser for the import command #########################################################################
    parser.add_argument('-i', '--openmvg',
                        help='path to openMVG directory, will automatically look for sfm_data, regions and pair files.')
    parser.add_argument('-s', '--sfm_data',
                        help='path to openMVG sfm_data data file.')
    parser.add_argument('-r', '--regions',
                        help='path to openMVG directory containing region files (feat, desc).')
    parser.add_argument('-m', '--matches',
                        help='path to openMVG matches file (eg. matches.f.txt')
    parser.add_argument('-o', '-k', '--kapture', required=True,
                        help='top directory where to save Kapture files.')
    parser.add_argument('--image_action', default='root_link', type=TransferAction,
                        help=f'''what to do with images:
        {TransferAction.root_link.name}: link to the root of the images directory (default) ;
        {TransferAction.link_absolute.name}: absolute individual file link ;
        {TransferAction.link_relative.name}: relative individual file link ;
        {TransferAction.copy.name}: copy file instead of creating link ;
        {TransferAction.move.name}: move file instead of creating link ;
        {TransferAction.skip.name}: do not create links
                                      ''')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # for debug, let kapture express itself.
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    logger.debug(f'{sys.argv[0]} \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()))

    # argument logic.
    if args.openmvg:
        args.openmvg = path.abspath(args.openmvg)
        # automatically look for sfm_data, regions and pair files if not specified
        if args.sfm_data is None:
            args.sfm_data = path.join(args.openmvg, OPENMVG_DEFAULT_JSON_FILE_NAME)
        if args.regions is None:
            args.regions = args.openmvg
        if args.matches is None:
            args.matches = path.join(args.openmvg, 'matches.f.txt')

    # normalizes paths
    if args.sfm_data is not None:
        args.sfm_data = path.normpath(path.abspath(args.sfm_data))
    if args.regions is not None:
        args.regions = path.normpath(path.abspath(args.regions))
    if args.matches is not None:
        args.matches = path.normpath(path.abspath(args.matches))
        assert args.matches.endswith('.txt')

    # sanity check
    if all(i is None for i in [args.sfm_data, args.regions, args.matches]):
        raise ValueError('Need openMVG files as input.')

    import_openmvg(args.sfm_data,
                   args.regions,
                   args.matches,
                   args.kapture,
                   args.image_action,
                   args.force)


if __name__ == '__main__':
    try:
        import_openmvg_command_line()
    except (AssertionError, LookupError, OSError):
        logger.error('Fatal error', exc_info=True)
