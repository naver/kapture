#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import os.path as path
import logging
import argparse

import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('copy_RobotCar_Seasons_images')


def kapture_copy_RobotCar_Seasons_images():
    parser = argparse.ArgumentParser(
        description='copy robotcar images to individual folders.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    parser.add_argument('--image-dir', required=True, help='path to robotcar images')
    parser.add_argument('--target', required=True, help='path to robotcar kapture root')

    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    imgdir = args.image_dir
    target = args.target
    for i in range(1, 50):
        loc_id = f"{i:02d}"

        logging.info(f'copying mapping images for location {i}')
        mapping_path = path.join(target, loc_id, 'mapping')
        if not path.isdir(mapping_path):
            continue
        try:
            kdata_mapping = kapture_from_dir(mapping_path)
            mapping_images = [f for _, _, f in kapture.flatten(kdata_mapping.records_camera)]
            import_record_data_from_dir_auto(imgdir, mapping_path, mapping_images, TransferAction.copy)
        except Exception as e:
            logging.warning(f'{mapping_path}, failed to copy images; error: {e}')

        logging.info(f'copying query images for location {i}')
        query_path = path.join(target, loc_id, 'query')
        if not path.isdir(query_path):
            continue
        try:
            kdata_query = kapture_from_dir(query_path)
            query_images = [f for _, _, f in kapture.flatten(kdata_query.records_camera)]
            import_record_data_from_dir_auto(imgdir, query_path, query_images, TransferAction.copy)
        except Exception as e:
            logging.warning(f'{query_path}, failed to copy images; error: {e}')


if __name__ == '__main__':
    kapture_copy_RobotCar_Seasons_images()
