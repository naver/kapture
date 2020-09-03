#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script builds a COLMAP model (map) from kapture format (images, cameras, trajectories: no features or matches!)
"""

import argparse
import logging
import os
import os.path as path
import shutil
from typing import List

import path_to_kapture  # noqa: F401
import kapture
import kapture.io.csv
import kapture.utils.logging
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.io.records import get_image_fullpath

import kapture.converter.colmap.colmap_command as colmap_lib
from kapture.converter.colmap.database import COLMAPDatabase
import kapture.converter.colmap.database_extra as database_extra

logger = logging.getLogger('colmap_build_sift_map')


def colmap_build_sift_map(kapture_path: str,
                          colmap_path: str,
                          colmap_binary: str,
                          colmap_use_cpu: bool,
                          colmap_gpu_index: str,
                          vocab_tree_path: str,
                          point_triangulator_options: List[str],
                          skip_list: List[str],
                          force: bool) -> None:
    """
    Build a colmap model using default SIFT features with the kapture data.

    :param kapture_path: path to the kapture to use
    :param colmap_path: path to the colmap build
    :param colmap_binary: path to the colmap executable
    :param colmap_use_cpu: to use cpu only (and ignore gpu) or to use also gpu
    :param colmap_gpu_index: gpu index for sift extractor and mapper
    :param vocab_tree_path: path to the colmap vocabulary tree file
    :param point_triangulator_options: options for the point triangulator
    :param skip_list: list of steps to skip
    :param force: Silently overwrite kapture files if already exists.
    """
    os.makedirs(colmap_path, exist_ok=True)

    # Load input files first to make sure it is OK
    logger.info('loading kapture files...')
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path)

    if not (kapture_data.records_camera and kapture_data.sensors):
        raise ValueError('records_camera, sensors are mandatory')
    if not kapture_data.trajectories:
        logger.info('there are no trajectories, running mapper instead of point_triangulator')

    if not os.path.isfile(vocab_tree_path):
        raise ValueError(f'Vocabulary Tree file does not exist: {vocab_tree_path}')

    # COLMAP does not fully support rigs.
    if kapture_data.rigs is not None and kapture_data.trajectories is not None:
        # make sure, rigs are not used in trajectories.
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

    # Set fixed name for COLMAP database
    colmap_db_path = path.join(colmap_path, 'colmap.db')
    image_list_path = path.join(colmap_path, 'images.list')
    reconstruction_path = path.join(colmap_path, "reconstruction")
    if 'delete_existing' not in skip_list:
        safe_remove_file(colmap_db_path, force)
        safe_remove_file(image_list_path, force)
        safe_remove_any_path(reconstruction_path, force)
    os.makedirs(reconstruction_path, exist_ok=True)

    if 'feature_extract' not in skip_list:
        logger.info("Step 1: Feature extraction using colmap")
        with open(image_list_path, 'w') as fid:
            for timestamp, sensor_id in sorted(kapture_data.records_camera.key_pairs()):
                fid.write(kapture_data.records_camera[timestamp][sensor_id] + "\n")

        colmap_lib.run_feature_extractor(
            colmap_binary,
            colmap_use_cpu,
            colmap_gpu_index,
            colmap_db_path,
            get_image_fullpath(kapture_path),
            image_list_path
        )

    # Update cameras in COLMAP:
    # - use only one camera for all images taken with the same camera (update all camera IDs)
    # - import camera intrinsics
    # - import camera pose
    if 'update_db_cameras' not in skip_list:
        logger.info("Step 2: Populate COLMAP DB with cameras and poses")
        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        database_extra.update_DB_cameras_and_poses(colmap_db, kapture_data)
        # close db before running colmap processes in order to avoid locks
        colmap_db.close()

    # Extract matches with COLMAP
    if 'matches' not in skip_list:
        logger.info("Step 3: Compute matches with colmap")

        colmap_lib.run_vocab_tree_matcher(
            colmap_binary,
            colmap_use_cpu,
            colmap_gpu_index,
            colmap_db_path,
            vocab_tree_path)

    if kapture_data.trajectories is not None:
        # Generate priors for reconstruction
        txt_path = path.join(colmap_path, "priors_for_reconstruction")
        os.makedirs(txt_path, exist_ok=True)
        if 'priors_for_reconstruction' not in skip_list:
            logger.info('Step 4: Exporting priors for reconstruction.')
            colmap_db = COLMAPDatabase.connect(colmap_db_path)
            database_extra.generate_priors_for_reconstruction(kapture_data, colmap_db, txt_path)
            colmap_db.close()

        # Point triangulator
        reconstruction_path = path.join(colmap_path, "reconstruction")
        os.makedirs(reconstruction_path, exist_ok=True)
        if 'triangulation' not in skip_list:
            logger.info("Step 5: Triangulation")
            colmap_lib.run_point_triangulator(
                colmap_binary,
                colmap_db_path,
                get_image_fullpath(kapture_path),
                txt_path,
                reconstruction_path,
                point_triangulator_options
            )
    else:
        # mapper
        reconstruction_path = path.join(colmap_path, "reconstruction")
        os.makedirs(reconstruction_path, exist_ok=True)
        if 'triangulation' not in skip_list:
            logger.info("Step 5: Triangulation")
            colmap_lib.run_mapper(
                colmap_binary,
                colmap_db_path,
                get_image_fullpath(kapture_path),
                None,
                reconstruction_path,
                point_triangulator_options
            )
            # use reconstruction 0 as main
            first_reconstruction = os.path.join(reconstruction_path, '0')
            files = os.listdir(first_reconstruction)
            for f in files:
                shutil.move(os.path.join(first_reconstruction, f), os.path.join(reconstruction_path, f))
            shutil.rmtree(first_reconstruction)

    # run model_converter
    if 'model_converter' not in skip_list:
        logger.info("Step 6: Export reconstruction results to txt")
        colmap_lib.run_model_converter(
            colmap_binary,
            reconstruction_path,
            reconstruction_path
        )


def colmap_build_sift_map_command_line():
    """
    Parse the command line arguments to build a colmap map using the given kapture data and the sift feature.
    """
    parser = argparse.ArgumentParser(description='create a Colmap model (map) from data specified in kapture format.'
                                                 'Only images and cameras are taken into account '
                                                 ' (no features or matches)')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const',
                                  dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='silently delete database if already exists.')
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory (colmap directory).')
    parser.add_argument('-colmap', '--colmap_binary', required=False,
                        default="colmap",
                        help='full path to colmap binary '
                             '(default is "colmap", i.e. assume the binary'
                             ' is in the user PATH).')
    parser.add_argument('--cpu', action='store_true',
                        default=False,
                        help='Ignore gpu and run on cpu only')
    parser.add_argument('-gpu_idx', '--colmap_gpu_index', required=False,
                        default=None,
                        help='GPU index used by colmap Sift extractor and matcher (e.g. "0" or "0,1,2").'
                             ' Default: None. Ignored if --cpu is used')
    parser.add_argument('-voc', '--vocab_tree_path', required=True,
                        help='full path to Vocabulary Tree file'
                             ' used for matching.')
    parser.add_argument('-s', '--skip', choices=['delete_existing',
                                                 'feature_extract',
                                                 'update_db_cameras',
                                                 'matches',
                                                 'priors_for_reconstruction',
                                                 'triangulation',
                                                 'model_converter'],
                        nargs='+', default=[],
                        help='steps to skip')

    args, point_triangulator_options = parser.parse_known_args()

    logger.setLevel(args.verbose)
    logging.getLogger('colmap').setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    args_dict['point_triangulator_options'] = point_triangulator_options
    logger.debug('colmap_build_sift_map.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    colmap_build_sift_map(args.input, args.output, args.colmap_binary,
                          args.cpu, args.colmap_gpu_index,
                          args.vocab_tree_path, point_triangulator_options,
                          args.skip, args.force)


if __name__ == '__main__':
    colmap_build_sift_map_command_line()
