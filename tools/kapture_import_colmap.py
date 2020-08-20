#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports data from a COLMAP database and/or reconstruction files

It will assume that if all rotation parameters are null, there is no prior rotation, and if all translation parameters
are null, there is no prior translation (using null values instead of zeros)
"""

import argparse
import logging
import os
import sys

import path_to_kapture  # enables import kapture
import kapture
import kapture.utils.logging
import kapture.io.csv
from kapture.io.records import TransferAction
from kapture.converter.colmap.import_colmap import import_colmap

logger = logging.getLogger('colmap')


def colmap_command_line() -> None:
    """
    Do the colmap import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='imports data from colmap database and/or text files.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete kapture if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-db', '--database',
                        help='path to COLMAP database file.')
    parser.add_argument('-txt', '--reconstruction',
                        help='path to COLMAP reconstruction triplet text file.')
    # images
    parser.add_argument('-im', '--images', default=None,
                        help='path to images directory.')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute],'
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-rig', '--rig',
                        help='path to COLMAP rig json file.')
    parser.add_argument('-o', '-k', '--kapture', required=True,
                        help='output directory where to save kapture files.')
    parser.add_argument('--skip_reconstruction', action='store_true', default=False,
                        help='skip the import of the kapture/reconstruction,'
                             'ie. Keypoints, Descriptors, Matches, Points3d, Observations.')
    parser.add_argument('--no_geometric_filtering', action='store_true', default=False,
                        help='don\'t remove matches failing geometric verification.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # for debug, let kapture express itself.
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(f'{sys.argv[0]} \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()))

    if args.image_transfer == TransferAction.root_link:
        logger.fatal('Root linking the images is not supported yet.')
        return
    logger.debug('cleaning output directory')
    os.makedirs(args.kapture, exist_ok=True)
    kapture.io.structure.delete_existing_kapture_files(args.kapture, force_erase=args.force)

    logger.info('importing colmap ...')
    kapture_data = import_colmap(
        kapture_dirpath=args.kapture,
        colmap_database_filepath=args.database,
        colmap_reconstruction_dirpath=args.reconstruction,
        colmap_images_dirpath=args.images,
        colmap_rig_filepath=args.rig,
        no_geometric_filtering=args.no_geometric_filtering,
        skip_reconstruction=args.skip_reconstruction,
        force_overwrite_existing=args.force,
        images_import_strategy=args.image_transfer if args.images is not None else TransferAction.skip
    )

    logger.info('saving to kapture  ...')
    logger.debug(f'\t"{args.kapture}"')
    kapture.io.csv.kapture_to_dir(args.kapture, kapture_data)


if __name__ == '__main__':
    colmap_command_line()
