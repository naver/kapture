#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports an openmvg file in the kapture format.
"""

import argparse
import logging
import sys

# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture.utils.logging
from kapture.io.records import TransferAction
# openmvg
from kapture.converter.openmvg.import_openmvg import import_openmvg


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
    parser.add_argument('-o', '--openmvg', required=True,
                        help='path to openMVG JSON file.')
    parser.add_argument('-k', '--kapture', required=True,
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

    # Check that we will not try to do symbolics links on Windows
    # because this is not possible today if the user is not admin
    if sys.platform.startswith("win") and \
            (args.image_link == TransferAction.link_relative or
             args.image_link == TransferAction.link_absolute or
             args.image_link == TransferAction.root_link):
        logger.fatal("It is currently impossible to link files on a Windows platform")
        logger.fatal(f"Please try another option for the image: either do nothing ({TransferAction.skip}),"
                     f" or {TransferAction.copy}")
        raise OSError("Image file linking not possible on Windows")
    else:
        import_openmvg(args.openmvg, args.kapture, args.image_action, args.force)


if __name__ == '__main__':
    try:
        import_openmvg_command_line()
    except (AssertionError, LookupError, OSError):
        logger.error('Fatal error', exc_info=True)
