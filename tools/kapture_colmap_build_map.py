#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script builds a COLMAP model (map) from kapture format (images, cameras, trajectories, features and matches!)
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
from kapture.io.records import get_image_fullpath

import kapture.converter.colmap.colmap_command as colmap_lib
from kapture.converter.colmap.database import COLMAPDatabase
import kapture.converter.colmap.database_extra as database_extra

logger = logging.getLogger('colmap_build_map')


def colmap_build_map(kapture_path: str,
                     colmap_path: str,
                     colmap_binary: str,
                     pairs_file_path: Optional[str],
                     use_colmap_matches_importer: bool,
                     point_triangulator_options: List[str],
                     skip_list: List[str],
                     force: bool) -> None:
    """
    Build a colmap model using custom features with the kapture data.

      :param kapture_path: path to the kapture to use
      :param colmap_path: path to the colmap build
      :param colmap_binary: path to the colmap executable
      :param pairs_file_path: Optional[str],
      :param use_colmap_matches_importer: bool,
      :param point_triangulator_options: options for the point triangulator
      :param skip_list: list of steps to skip
      :param force: Silently overwrite kapture files if already exists.
      """
    # Load input files first to make sure it is OK
    logger.info('loading kapture files...')
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path, pairs_file_path)

    colmap_build_map_from_loaded_data(kapture_data,
                                      kapture_path,
                                      colmap_path,
                                      colmap_binary,
                                      use_colmap_matches_importer,
                                      point_triangulator_options,
                                      skip_list,
                                      force)


def colmap_build_map_from_loaded_data(kapture_data: kapture.Kapture,
                                      kapture_path: str,
                                      colmap_path: str,
                                      colmap_binary: str,
                                      use_colmap_matches_importer: bool,
                                      point_triangulator_options: List[str],
                                      skip_list: List[str],
                                      force: bool) -> None:
    """
    Build a colmap model using custom features with the kapture data.

    :param kapture_data: kapture data to use
    :param kapture_path: path to the kapture to use
    :param colmap_path: path to the colmap build
    :param colmap_binary: path to the colmap executable
    :param use_colmap_matches_importer: bool,
    :param point_triangulator_options: options for the point triangulator
    :param skip_list: list of steps to skip
    :param force: Silently overwrite kapture files if already exists.
    """
    os.makedirs(colmap_path, exist_ok=True)

    if not (kapture_data.records_camera and kapture_data.sensors and kapture_data.keypoints and kapture_data.matches):
        raise ValueError('records_camera, sensors, keypoints, matches are mandatory')
    if not kapture_data.trajectories:
        logger.info('there are no trajectories, running mapper instead of point_triangulator')

    # COLMAP does not fully support rigs.
    if kapture_data.rigs is not None and kapture_data.trajectories is not None:
        # make sure, rigs are not used in trajectories.
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

    # Set fixed name for COLMAP database
    colmap_db_path = path.join(colmap_path, 'colmap.db')
    reconstruction_path = path.join(colmap_path, "reconstruction")
    priors_txt_path = path.join(colmap_path, "priors_for_reconstruction")
    if 'delete_existing' not in skip_list:
        safe_remove_file(colmap_db_path, force)
        safe_remove_any_path(reconstruction_path, force)
        safe_remove_any_path(priors_txt_path, force)
    os.makedirs(reconstruction_path, exist_ok=True)

    if 'colmap_db' not in skip_list:
        logger.info('Using precomputed keypoints and matches')
        logger.info('Step 1: Export kapture format to colmap')

        colmap_db = COLMAPDatabase.connect(colmap_db_path)
        if kapture_data.descriptors is not None:
            kapture_data.descriptors.clear()
        database_extra.kapture_to_colmap(kapture_data, kapture_path, colmap_db,
                                         export_two_view_geometry=not use_colmap_matches_importer)
        # close db before running colmap processes in order to avoid locks
        colmap_db.close()

        if use_colmap_matches_importer:
            logger.info('Step 2: Run geometric verification')
            logger.debug('running colmap matches_importer...')
            colmap_lib.run_matches_importer_from_kapture(
                colmap_binary,
                colmap_use_cpu=True,
                colmap_gpu_index=None,
                colmap_db_path=colmap_db_path,
                kapture_data=kapture_data,
                force=force
            )
        else:
            logger.info('Step 2: Run geometric verification - skipped')

    if kapture_data.trajectories is not None:
        # Generate priors for reconstruction
        os.makedirs(priors_txt_path, exist_ok=True)
        if 'priors_for_reconstruction' not in skip_list:
            logger.info('Step 3: Exporting priors for reconstruction.')
            colmap_db = COLMAPDatabase.connect(colmap_db_path)
            database_extra.generate_priors_for_reconstruction(kapture_data, colmap_db, priors_txt_path)
            colmap_db.close()

        # Point triangulator
        reconstruction_path = path.join(colmap_path, "reconstruction")
        os.makedirs(reconstruction_path, exist_ok=True)
        if 'triangulation' not in skip_list:
            logger.info("Step 4: Triangulation")
            colmap_lib.run_point_triangulator(
                colmap_binary,
                colmap_db_path,
                get_image_fullpath(kapture_path),
                priors_txt_path,
                reconstruction_path,
                point_triangulator_options
            )
    else:
        # mapper
        reconstruction_path = path.join(colmap_path, "reconstruction")
        os.makedirs(reconstruction_path, exist_ok=True)
        if 'triangulation' not in skip_list:
            logger.info("Step 4: Triangulation")
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
        logger.info("Step 5: Export reconstruction results to txt")
        colmap_lib.run_model_converter(
            colmap_binary,
            reconstruction_path,
            reconstruction_path
        )


def colmap_build_map_command_line():
    """
    Parse the command line arguments to build a colmap map using the given kapture data.
    """
    parser = argparse.ArgumentParser(description=('create a Colmap model (map) from data specified in kapture format.'
                                                  'kapture data must contain keypoints and matches'))
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
    parser.add_argument('--use-colmap-matches-importer', action='store_true', default=False,
                        help='Use colmap matches importer instead of manually filling the two_view_geometry table')
    parser.add_argument('--pairs-file-path',
                        default=None,
                        type=str,
                        help=('text file in the csv format; where each line is image_name1, image_name2, score '
                              'which contains the image pairs to match, can be used to filter loaded matches'))
    parser.add_argument('-s', '--skip', choices=['delete_existing',
                                                 'colmap_db',
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
    logger.debug('colmap_build_map.py \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v)) for k, v in args_dict.items()))
    colmap_build_map(args.input, args.output, args.colmap_binary,
                     args.pairs_file_path, args.use_colmap_matches_importer, point_triangulator_options,
                     args.skip, args.force)


if __name__ == '__main__':
    colmap_build_map_command_line()
