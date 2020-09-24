#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script localize images on an existing COLMAP model (map) from kapture format
images, cameras, trajectories: no features or matches!
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

logger = logging.getLogger('colmap_localize_sift')


def colmap_localize_sift(kapture_path: str,
                         colmap_path: str,
                         input_database_path: str,
                         input_reconstruction_path: str,
                         colmap_binary: str,
                         colmap_use_cpu: bool,
                         colmap_gpu_index: str,
                         vocab_tree_path: str,
                         image_registrator_options: List[str],
                         skip_list: List[str],
                         force: bool) -> None:
    """
    Localize images on a colmap model using default SIFT features with the kapture data.

    :param kapture_path: path to the kapture to use
    :param colmap_path: path to the colmap build
    :param input_database_path: path to the map colmap.db
    :param input_database_path: path to the map colmap.db
    :param input_reconstruction_path: path to the map reconstruction folder
    :param colmap_binary: path to the colmap binary executable
    :param colmap_use_cpu: to use cpu only (and ignore gpu) or to use also gpu
    :param colmap_gpu_index: gpu index for sift extractor and mapper
    :param vocab_tree_path: path to the colmap vocabulary tree file
    :param image_registrator_options: options for the image registrator
    :param skip_list: list of steps to skip
    :param force: Silently overwrite kapture files if already exists.
    """
    os.makedirs(colmap_path, exist_ok=True)
    # Set fixed name for COLMAP database

    # Load input files first to make sure it is OK
    logger.info('loading kapture files...')
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path)

    if not (kapture_data.records_camera and kapture_data.sensors):
        raise ValueError('records_camera, sensors are mandatory')

    if kapture_data.trajectories:
        logger.warning("Input data contains trajectories: they will be ignored")
        kapture_data.trajectories.clear()
    else:
        kapture_data.trajectories = kapture.Trajectories()

    if not os.path.isfile(vocab_tree_path):
        raise ValueError(f'Vocabulary Tree file does not exist: {vocab_tree_path}')

    # COLMAP does not fully support rigs.
    if kapture_data.rigs is not None and kapture_data.trajectories is not None:
        # make sure, rigs are not used in trajectories.
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

    # Prepare output
    # Set fixed name for COLMAP database
    colmap_db_path = path.join(colmap_path, 'colmap.db')
    image_list_path = path.join(colmap_path, 'images.list')
    reconstruction_path = path.join(colmap_path, "reconstruction")
    if 'delete_existing' not in skip_list:
        safe_remove_file(colmap_db_path, force)
        safe_remove_file(image_list_path, force)
        safe_remove_any_path(reconstruction_path, force)
    os.makedirs(reconstruction_path, exist_ok=True)

    # Copy colmap db to output
    if not os.path.exists(colmap_db_path):
        shutil.copy(input_database_path, colmap_db_path)

    # find correspondences between the colmap db and the kapture data
    images_all = {image_path: (ts, cam_id)
                  for ts, shot in kapture_data.records_camera.items()
                  for cam_id, image_path in shot.items()}

    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    colmap_image_ids = database_extra.get_colmap_image_ids_from_db(colmap_db)
    colmap_cameras = database_extra.get_camera_ids_from_database(colmap_db)
    colmap_images = database_extra.get_images_from_database(colmap_db)
    colmap_db.close()

    # dict ( kapture_camera -> colmap_camera_id )
    colmap_camera_ids = {images_all[image_path][1]: colmap_cam_id
                         for image_path, colmap_cam_id in colmap_images if image_path in images_all}

    images_to_add = {image_path: value
                     for image_path, value in images_all.items()
                     if image_path not in colmap_image_ids}

    flatten_images_to_add = [(ts, kapture_cam_id, image_path)
                             for image_path, (ts, kapture_cam_id) in images_to_add.items()]

    if 'feature_extract' not in skip_list:
        logger.info("Step 1: Feature extraction using colmap")
        with open(image_list_path, 'w') as fid:
            for image in images_to_add.keys():
                fid.write(image + "\n")

        colmap_lib.run_feature_extractor(
            colmap_binary,
            colmap_use_cpu,
            colmap_gpu_index,
            colmap_db_path,
            get_image_fullpath(kapture_path),
            image_list_path
        )

    if 'matches' not in skip_list:
        logger.info("Step 2: Compute matches with colmap")
        colmap_lib.run_vocab_tree_matcher(
            colmap_binary,
            colmap_use_cpu,
            colmap_gpu_index,
            colmap_db_path,
            vocab_tree_path,
            image_list_path
        )

    if 'fix_db_cameras' not in skip_list:
        logger.info("Step 3: Replace colmap generated cameras with kapture cameras")
        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        database_extra.foreign_keys_off(colmap_db)

        # remove colmap generated cameras
        after_feature_extraction_colmap_cameras = database_extra.get_camera_ids_from_database(colmap_db)
        colmap_cameras_to_remove = [cam_id
                                    for cam_id in after_feature_extraction_colmap_cameras
                                    if cam_id not in colmap_cameras]
        for cam_id in colmap_cameras_to_remove:
            database_extra.remove_camera(colmap_db, cam_id)

        # put the correct cameras and image extrinsic back into the database
        cameras_to_add = kapture.Sensors()
        for image_path, (ts, kapture_cam_id) in images_to_add.items():
            if kapture_cam_id not in colmap_camera_ids:
                kapture_cam = kapture_data.sensors[kapture_cam_id]
                cameras_to_add[kapture_cam_id] = kapture_cam
        colmap_added_camera_ids = database_extra.add_cameras_to_database(cameras_to_add, colmap_db)
        colmap_camera_ids.update(colmap_added_camera_ids)

        database_extra.update_images_in_database_from_flatten(
            colmap_db,
            flatten_images_to_add,
            kapture_data.trajectories,
            colmap_camera_ids
        )

        database_extra.foreign_keys_on(colmap_db)
        colmap_db.commit()
        colmap_db.close()

    if 'image_registrator' not in skip_list:
        logger.info("Step 4: Run image_registrator")
        # run image_registrator
        colmap_lib.run_image_registrator(
            colmap_binary,
            colmap_db_path,
            input_reconstruction_path,
            reconstruction_path,
            image_registrator_options
        )

    # run model_converter
    if 'model_converter' not in skip_list:
        logger.info("Step 5: Export reconstruction results to txt")
        colmap_lib.run_model_converter(
            colmap_binary,
            reconstruction_path,
            reconstruction_path
        )


def colmap_localize_sift_command_line():
    """
    Parse the command line arguments to localize images on an existing colmap map using the given kapture data.
    """
    parser = argparse.ArgumentParser(description=('localize images on a colmap model (map) '
                                                  'from data specified in kapture format.'
                                                  'Only images and cameras are taken into account '
                                                  ' (no features or matches)'))
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
    parser.add_argument('-db', '--database',
                        help='path to COLMAP database file.')
    parser.add_argument('-txt', '--reconstruction',
                        help='path to COLMAP reconstruction triplet text file.')
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
                             ' Default: None means all available gpus. Ignored if --cpu is used')
    parser.add_argument('-voc', '--vocab_tree_path', required=True,
                        help='full path to Vocabulary Tree file'
                             ' used for matching.')
    parser.add_argument('-s', '--skip', choices=['delete_existing',
                                                 'feature_extract'
                                                 'matches',
                                                 'fix_db_cameras',
                                                 'image_registrator',
                                                 'model_converter'],
                        nargs='+', default=[],
                        help='steps to skip')

    args, image_registrator_options = parser.parse_known_args()

    logger.setLevel(args.verbose)
    logging.getLogger('colmap').setLevel(args.verbose)
    if args.verbose <= logging.INFO:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    args_dict = vars(args)
    args_dict['image_registrator_options'] = image_registrator_options
    logger.debug('colmap_localize_sift.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    colmap_localize_sift(args.input, args.output,
                         args.database, args.reconstruction,
                         args.colmap_binary,
                         args.cpu, args.colmap_gpu_index,
                         args.vocab_tree_path, image_registrator_options,
                         args.skip, args.force)


if __name__ == '__main__':
    colmap_localize_sift_command_line()
