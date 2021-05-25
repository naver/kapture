#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Rename features inside kapture data
"""

import logging
import os
import os.path as path
import argparse
import shutil

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.features
import kapture.io.csv

logger = logging.getLogger('rename_features')


def rename_features(kapture_dirpath: str,
                    feature_type: str,
                    current_name: str,
                    new_name: str) -> None:
    """
    Rename features inside kapture data.
    """
    with kapture.io.csv.get_all_tar_handlers(kapture_dirpath) as tar_handlers:
        kapture_data = kapture.io.csv.kapture_from_dir(kapture_dirpath, tar_handlers=tar_handlers)
    if feature_type == 'global_features':
        # rename global_features folder
        if kapture_data.global_features is None or current_name not in kapture_data.global_features:
            raise ValueError(f'couldn\'t find {feature_type} in '
                             f'kapture_data.global_features: {kapture_data.global_features}')
        inpath = kapture.io.features.get_features_fullpath(kapture.GlobalFeatures, current_name, kapture_dirpath)
        outpath = kapture.io.features.get_features_fullpath(kapture.GlobalFeatures, new_name, kapture_dirpath)
        if path.isdir(outpath):
            raise ValueError(f'{outpath} already exists')
        os.rename(inpath, outpath)
    elif feature_type == 'descriptors':
        # rename descriptors folder
        if kapture_data.descriptors is None or current_name not in kapture_data.descriptors:
            raise ValueError(f'couldn\'t find {feature_type} in '
                             f'kapture_data.descriptors: {kapture_data.descriptors}')
        inpath = kapture.io.features.get_features_fullpath(kapture.Descriptors, current_name, kapture_dirpath)
        outpath = kapture.io.features.get_features_fullpath(kapture.Descriptors, new_name, kapture_dirpath)
        if path.isdir(outpath):
            raise ValueError(f'{outpath} already exists')
        os.rename(inpath, outpath)
    elif feature_type == 'keypoints':
        # rename keypoints folder
        if kapture_data.keypoints is not None and current_name in kapture_data.keypoints:
            inpath = kapture.io.features.get_features_fullpath(kapture.Keypoints, current_name, kapture_dirpath)
            outpath = kapture.io.features.get_features_fullpath(kapture.Keypoints, new_name, kapture_dirpath)
            if path.isdir(outpath):
                raise ValueError(f'{outpath} already exists')
            os.rename(inpath, outpath)
        else:
            logger.warning(f'couldn\'t find {feature_type} in '
                           f'kapture_data.keypoints: {kapture_data.keypoints}')

        # rename matches folder
        if kapture_data.matches is not None and current_name in kapture_data.matches:
            inpath = kapture.io.features.get_features_fullpath(kapture.Matches, current_name, kapture_dirpath)
            outpath = kapture.io.features.get_features_fullpath(kapture.Matches, new_name, kapture_dirpath)
            if path.isdir(outpath):
                raise ValueError(f'{outpath} already exists')
            os.rename(inpath, outpath)
        else:
            logger.debug(f'couldn\'t find {feature_type} in '
                         f'kapture_data.matches: {kapture_data.matches}')

        # update descriptors .txt files that uses this keypoints
        if kapture_data.descriptors is not None:
            for descriptors_type, kapture_descriptors in kapture_data.descriptors.items():
                if kapture_descriptors.keypoints_type == current_name:
                    logger.debug(f'descriptors {descriptors_type}, changing keypoints_type')
                    descriptors_csv_path = kapture.io.csv.get_feature_csv_fullpath(kapture.Descriptors,
                                                                                   descriptors_type,
                                                                                   kapture_dirpath)
                    out_kapture_descriptors = kapture.Descriptors(kapture_descriptors.type_name,
                                                                  kapture_descriptors.dtype,
                                                                  kapture_descriptors.dsize,
                                                                  new_name,
                                                                  kapture_descriptors.metric_type,
                                                                  kapture_descriptors)
                    kapture.io.csv.descriptors_to_file(descriptors_csv_path, out_kapture_descriptors)

        # update observations.txt
        if kapture_data.observations is not None:
            if kapture_data.keypoints is None or current_name not in kapture_data.keypoints:
                raise ValueError(f'{current_name} not found in keypoints, perhaps a previous attempt failed midway'
                                 'observations won\'t load properly in this state so fix them manually')
            new_observations = kapture.Observations()
            for point3d_idx, keypoints_type in kapture_data.observations.key_pairs():
                output_keypoints_type = keypoints_type
                if output_keypoints_type == current_name:
                    output_keypoints_type = new_name
                observation_array = kapture_data.observations.get(point3d_idx)[keypoints_type]
                new_observations.setdefault(point3d_idx, {})[output_keypoints_type] = observation_array
            observations_csv_path = kapture.io.csv.get_csv_fullpath(kapture.Observations, kapture_dirpath)
            observations_csv_path_bak = observations_csv_path + '.rename_bak'
            if path.isfile(observations_csv_path_bak):
                os.remove(observations_csv_path_bak)
            shutil.copy(observations_csv_path, observations_csv_path_bak)
            kapture.io.csv.observations_to_file(observations_csv_path, new_observations)
    else:
        raise ValueError(f'feature_type {feature_type} is unknown')
    logger.debug('all done!')


def rename_features_command_line() -> None:
    """
    Rename features inside kapture data.
    """
    parser = argparse.ArgumentParser(
        description='rename features inside kapture data')
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
    parser.add_argument('--feature-type', required=True, choices=['keypoints', 'descriptors', 'global_features'],
                        help='types of features to rename.')
    parser.add_argument('--current-name', required=True,
                        help='keypoints_type, descriptors_type, global_features_type depending on selected features.')
    parser.add_argument('--new-name', required=True,
                        help='keypoints_type, descriptors_type, global_features_type depending on selected features.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    rename_features(args.input, args.feature_type, args.current_name, args.new_name)


if __name__ == '__main__':
    rename_features_command_line()
