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
from typing import List, Optional

import path_to_kapture  # noqa: F401
import kapture
import kapture.io.csv
import kapture.utils.logging
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
from kapture.core.Trajectories import rigs_remove_inplace

import kapture.converter.colmap.colmap_command as colmap_lib
from kapture.converter.colmap.database import COLMAPDatabase
import kapture.converter.colmap.database_extra as database_extra

logger = logging.getLogger('colmap_localize')


def colmap_localize(kapture_path: str,
                    colmap_path: str,
                    input_database_path: str,
                    input_reconstruction_path: str,
                    colmap_binary: str,
                    pairs_file_path: Optional[str],
                    use_colmap_matches_importer: bool,
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
    :param pairs_file_path: Optional[str],
    :param use_colmap_matches_importer: bool,
    :param image_registrator_options: options for the image registrator
    :param skip_list: list of steps to skip
    :param force: Silently overwrite kapture files if already exists.
    """
    # Load input files first to make sure it is OK
    logger.info('loading kapture files...')
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path, pairs_file_path)

    colmap_localize_from_loaded_data(kapture_data,
                                     kapture_path,
                                     colmap_path,
                                     input_database_path,
                                     input_reconstruction_path,
                                     colmap_binary,
                                     use_colmap_matches_importer,
                                     image_registrator_options,
                                     skip_list,
                                     force)


def colmap_localize_from_loaded_data(kapture_data: kapture.Kapture,
                                     kapture_path: str,
                                     colmap_path: str,
                                     input_database_path: str,
                                     input_reconstruction_path: str,
                                     colmap_binary: str,
                                     use_colmap_matches_importer: bool,
                                     image_registrator_options: List[str],
                                     skip_list: List[str],
                                     force: bool) -> None:
    """
    Localize images on a colmap model using default SIFT features with the kapture data.

    :param kapture_data: kapture data to use
    :param kapture_path: path to the kapture to use
    :param colmap_path: path to the colmap build
    :param input_database_path: path to the map colmap.db
    :param input_database_path: path to the map colmap.db
    :param input_reconstruction_path: path to the map reconstruction folder
    :param colmap_binary: path to the colmap binary executable
    :param use_colmap_matches_importer: bool,
    :param image_registrator_options: options for the image registrator
    :param skip_list: list of steps to skip
    :param force: Silently overwrite kapture files if already exists.
    """
    os.makedirs(colmap_path, exist_ok=True)

    if not (kapture_data.records_camera and kapture_data.sensors and kapture_data.keypoints and kapture_data.matches):
        raise ValueError('records_camera, sensors, keypoints, matches are mandatory')

    if kapture_data.trajectories:
        logger.warning("Input data contains trajectories: they will be ignored")
        kapture_data.trajectories.clear()
    else:
        kapture_data.trajectories = kapture.Trajectories()

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

    if 'import_to_db' not in skip_list:
        logger.info("Step 1: Add precomputed keypoints and matches to colmap db")
        cameras_to_add = kapture.Sensors()
        for _, (_, kapture_cam_id) in images_to_add.items():
            if kapture_cam_id not in colmap_camera_ids:
                kapture_cam = kapture_data.sensors[kapture_cam_id]
                cameras_to_add[kapture_cam_id] = kapture_cam
        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        colmap_added_camera_ids = database_extra.add_cameras_to_database(cameras_to_add, colmap_db)
        colmap_camera_ids.update(colmap_added_camera_ids)

        colmap_added_image_ids = database_extra.add_images_to_database_from_flatten(
            colmap_db, flatten_images_to_add, kapture_data.trajectories, colmap_camera_ids)
        colmap_image_ids.update(colmap_added_image_ids)

        colmap_image_ids_reversed = {v: k for k, v in colmap_image_ids.items()}  # colmap_id : name

        # add new features
        colmap_keypoints = database_extra.get_keypoints_set_from_database(colmap_db, colmap_image_ids_reversed)

        keypoints_all = kapture_data.keypoints
        keypoints_to_add = {name for name in keypoints_all if name not in colmap_keypoints}
        keypoints_to_add = kapture.Keypoints(keypoints_all.type_name, keypoints_all.dtype, keypoints_all.dsize,
                                             keypoints_to_add)
        database_extra.add_keypoints_to_database(colmap_db, keypoints_to_add, kapture_path, colmap_image_ids)

        # add new matches
        colmap_matches = kapture.Matches(database_extra.get_matches_set_from_database(colmap_db,
                                                                                      colmap_image_ids_reversed))
        colmap_matches.normalize()

        matches_all = kapture_data.matches
        matches_to_add = kapture.Matches({pair for pair in matches_all if pair not in colmap_matches})
        # print(list(matches_to_add))
        database_extra.add_matches_to_database(colmap_db, matches_to_add, kapture_path, colmap_image_ids,
                                               export_two_view_geometry=not use_colmap_matches_importer)
        colmap_db.close()

    if use_colmap_matches_importer:
        logger.info('Step 2: Run geometric verification')
        logger.debug('running colmap matches_importer...')
        # compute two view geometry
        colmap_lib.run_matches_importer_from_kapture(
            colmap_binary,
            colmap_use_cpu=True,
            colmap_gpu_index=None,
            colmap_db_path=colmap_db_path,
            kapture_data=kapture_data,
            force=force)
    else:
        logger.info('Step 2: Run geometric verification - skipped')
    if 'image_registrator' not in skip_list:
        logger.info("Step 3: Run image_registrator")
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
        logger.info("Step 4: Export reconstruction results to txt")
        colmap_lib.run_model_converter(
            colmap_binary,
            reconstruction_path,
            reconstruction_path
        )


def colmap_localize_command_line():
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
    parser.add_argument('--use-colmap-matches-importer', action='store_true', default=False,
                        help='Use colmap matches_importer instead of manually filling the two_view_geometry table')
    parser.add_argument('--pairs-file-path',
                        default=None,
                        type=str,
                        help=('text file in the csv format; where each line is image_name1, image_name2, score '
                              'which contains the image pairs to match, can be used to filter loaded matches'))
    parser.add_argument('-s', '--skip', choices=['delete_existing',
                                                 'import_to_db',
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
    logger.debug('colmap_localize.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))

    colmap_localize(args.input, args.output,
                    args.database, args.reconstruction,
                    args.colmap_binary,
                    args.pairs_file_path, args.use_colmap_matches_importer,
                    image_registrator_options,
                    args.skip, args.force)


if __name__ == '__main__':
    colmap_localize_command_line()
