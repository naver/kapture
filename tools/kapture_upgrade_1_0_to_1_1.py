#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data in version 1.0 to version 1.1
"""

import logging
import os
import os.path as path
import argparse
from typing import Optional
import shutil
# import numpy as np like in kapture.io.csv
# so that types written as "np.float32" are understood by read_old_image_features_csv
import numpy as np  # noqa: F401

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
import kapture.io.features
import kapture.io.csv
from kapture.io.records import import_record_data_from_dir_auto
from kapture.utils.paths import populate_files_in_dirpath
from kapture.io.binary import TransferAction
from kapture.utils.upgrade import CSV_FILENAMES_1_0, read_old_image_features_csv

logger = logging.getLogger('upgrade_1_0_to_1_1')


def upgrade_1_0_to_1_1(kapture_dirpath: str,  # noqa: C901: function a bit long but well documented
                       output_path: str,
                       keypoints_type: Optional[str],
                       descriptors_type: Optional[str],
                       global_features_type: Optional[str],
                       descriptors_metric_type: str,
                       global_features_metric_type: str,
                       images_import_strategy: TransferAction,
                       force_overwrite_existing: bool) -> None:
    """
    Convert kapture data in version 1.0 to version 1.1
    """
    os.makedirs(output_path, exist_ok=True)
    delete_existing_kapture_files(output_path, force_erase=force_overwrite_existing)
    # some text files didn't change, just change their header
    os.makedirs(path.join(output_path, 'sensors'), exist_ok=True)
    os.makedirs(path.join(output_path, 'reconstruction'), exist_ok=True)
    for csv_filename in CSV_FILENAMES_1_0:
        csv_fullpath = path.join(kapture_dirpath, csv_filename)
        if path.isfile(csv_fullpath):
            logger.debug(f'converting {csv_fullpath}...')
            old_version = kapture.io.csv.get_version_from_csv_file(csv_fullpath)
            if 'points3d' not in csv_filename:
                assert old_version == '1.0'
            else:
                assert (old_version is None or old_version == '1.0')
            csv_output_path = path.join(output_path, csv_filename)
            with open(csv_fullpath, 'r') as source_file:
                with open(csv_output_path, 'w') as output_file:
                    if old_version is not None:
                        source_file.readline()  # read and ignore header
                    # write replacement header
                    output_file.write(kapture.io.csv.KAPTURE_FORMAT_1 + kapture.io.csv.kapture_linesep)
                    shutil.copyfileobj(source_file, output_file)

    # keypoints
    keypoints_dir_path = path.join(kapture_dirpath, 'reconstruction', 'keypoints')
    keypoints_csv_path = path.join(keypoints_dir_path, 'keypoints.txt')
    if path.isdir(keypoints_dir_path) and path.isfile(keypoints_csv_path):
        logger.debug(f'converting {keypoints_dir_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(keypoints_csv_path)
        assert old_version == '1.0'
        name, dtype, dsize = read_old_image_features_csv(keypoints_csv_path)
        if keypoints_type is None:
            assert name != ''
            keypoints_type = name
        keypoints = kapture.Keypoints(name, dtype, dsize)
        keypoints_csv_output_path = path.join(output_path,
                                              kapture.io.csv.FEATURES_CSV_FILENAMES[kapture.Keypoints](keypoints_type))
        keypoints_output_dir = path.dirname(keypoints_csv_output_path)
        kapture.io.csv.keypoints_to_file(keypoints_csv_output_path, keypoints)
        # now copy all .kpt files
        keypoints_filenames = populate_files_in_dirpath(keypoints_dir_path, '.kpt')
        for keypoints_filename in keypoints_filenames:
            keypoints_output_file = path.join(keypoints_output_dir, keypoints_filename)
            os.makedirs(path.dirname(keypoints_output_file), exist_ok=True)
            shutil.copy(path.join(keypoints_dir_path, keypoints_filename), keypoints_output_file)

    # descriptors
    descriptors_dir_path = path.join(kapture_dirpath, 'reconstruction', 'descriptors')
    descriptors_csv_path = path.join(descriptors_dir_path, 'descriptors.txt')
    if path.isdir(descriptors_dir_path) and path.isfile(descriptors_csv_path):
        logger.debug(f'converting {descriptors_dir_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(descriptors_csv_path)
        assert old_version == '1.0'
        assert keypoints_type is not None
        name, dtype, dsize = read_old_image_features_csv(descriptors_csv_path)
        if descriptors_type is None:
            assert name != ''
            descriptors_type = name
        descriptors = kapture.Descriptors(name, dtype, dsize, keypoints_type, descriptors_metric_type)
        descriptors_csv_output_path = path.join(output_path,
                                                kapture.io.csv.FEATURES_CSV_FILENAMES[kapture.Descriptors](
                                                    descriptors_type)
                                                )
        descriptors_output_dir = path.dirname(descriptors_csv_output_path)
        kapture.io.csv.descriptors_to_file(descriptors_csv_output_path, descriptors)
        # now copy all .desc files
        descriptors_filenames = populate_files_in_dirpath(descriptors_dir_path, '.desc')
        for descriptors_filename in descriptors_filenames:
            descriptors_output_file = path.join(descriptors_output_dir, descriptors_filename)
            os.makedirs(path.dirname(descriptors_output_file), exist_ok=True)
            shutil.copy(path.join(descriptors_dir_path, descriptors_filename), descriptors_output_file)

    # matches
    matches_dir_path = path.join(kapture_dirpath, 'reconstruction', 'matches')
    if path.isdir(matches_dir_path):
        logger.debug(f'converting {matches_dir_path}...')
        assert keypoints_type is not None
        matches_output_dir = kapture.io.features.get_matches_fullpath(None, keypoints_type, output_path)
        # now copy all .matches files
        matches_filenames = populate_files_in_dirpath(matches_dir_path, '.matches')
        for matches_filename in matches_filenames:
            matches_output_file = path.join(matches_output_dir, matches_filename)
            os.makedirs(path.dirname(matches_output_file), exist_ok=True)
            shutil.copy(path.join(matches_dir_path, matches_filename), matches_output_file)

    # global features
    global_features_dir_path = path.join(kapture_dirpath, 'reconstruction', 'global_features')
    global_features_csv_path = path.join(global_features_dir_path, 'global_features.txt')
    if path.isdir(global_features_dir_path) and path.isfile(global_features_csv_path):
        logger.debug(f'converting {global_features_dir_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(global_features_csv_path)
        assert old_version == '1.0'
        assert keypoints_type is not None
        name, dtype, dsize = read_old_image_features_csv(global_features_csv_path)
        if global_features_type is None:
            assert name != ''
            global_features_type = name
        global_features = kapture.GlobalFeatures(name, dtype, dsize, global_features_metric_type)
        global_features_csv_output_path = path.join(output_path,
                                                    kapture.io.csv.FEATURES_CSV_FILENAMES[kapture.GlobalFeatures](
                                                        global_features_type)
                                                    )
        global_features_output_dir = path.dirname(global_features_csv_output_path)
        kapture.io.csv.global_features_to_file(global_features_csv_output_path, global_features)
        # now copy all .gfeat files
        global_features_filenames = populate_files_in_dirpath(global_features_dir_path, '.gfeat')
        for global_features_filename in global_features_filenames:
            global_features_output_file = path.join(global_features_output_dir, global_features_filename)
            os.makedirs(path.dirname(global_features_output_file), exist_ok=True)
            shutil.copy(path.join(global_features_dir_path, global_features_filename), global_features_output_file)

    # observations
    observations_csv_filename = path.join('reconstruction', 'observations.txt')
    observations_csv_path = path.join(kapture_dirpath, observations_csv_filename)
    if path.isfile(observations_csv_path):
        logger.debug(f'converting {observations_csv_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(observations_csv_path)
        assert old_version == '1.0'
        assert keypoints_type is not None
        observations = kapture.Observations()
        with open(observations_csv_path, 'r') as source_file:
            table = kapture.io.csv.table_from_file(source_file)
            # point3d_id, [image_path, feature_id]*
            for points3d_id_str, *pairs in table:
                points3d_id = int(points3d_id_str)
                if len(pairs) > 1:
                    image_paths = pairs[0::2]
                    keypoints_ids = pairs[1::2]
                    for image_path, keypoint_id in zip(image_paths, keypoints_ids):
                        observations.add(points3d_id, keypoints_type, image_path, int(keypoint_id))
        observations_output_path = path.join(output_path, observations_csv_filename)
        kapture.io.csv.observations_to_file(observations_output_path, observations)

    # records_data
    records_data_path = path.join(kapture_dirpath, 'sensors', 'records_data')
    logger.debug(f'converting {records_data_path}...')
    filename_list = list(populate_files_in_dirpath(records_data_path))
    import_record_data_from_dir_auto(records_data_path, output_path, filename_list, images_import_strategy)
    logger.debug('all done!')


def upgrade_1_0_to_1_1_command_line() -> None:
    """
    Convert kapture data in version 1.0 to version 1.1.
    """
    parser = argparse.ArgumentParser(
        description='convert kapture data in version 1.0 to version 1.1')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # export ###########################################################################################################
    parser.add_argument('-i', '--input', required=True, help='input path to kapture directory')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    parser.add_argument('--keypoints-type', default=None,
                        help='types of keypoints.')
    parser.add_argument('--descriptors-type', default=None, help='types of descriptors.')
    parser.add_argument('--descriptors-metric-type', default='L2', help='types of descriptors.')
    parser.add_argument('--global-features-type', default=None,
                        help='types of global features.')
    parser.add_argument('--global-features-metric-type', default='L2', help='types of descriptors.')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.skip,
                        help=f'How to import images [skip], '
                        f'choose among: {", ".join(a.name for a in TransferAction)}')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    upgrade_1_0_to_1_1(args.input, args.output,
                       args.keypoints_type, args.descriptors_type, args.global_features_type,
                       args.descriptors_metric_type, args.global_features_metric_type,
                       args.image_transfer,
                       args.force)


if __name__ == '__main__':
    upgrade_1_0_to_1_1_command_line()
