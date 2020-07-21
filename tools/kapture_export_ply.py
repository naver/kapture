#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to plot a kapture to a PLY file.
"""

import argparse
import logging
import os
import os.path as path
from tqdm import tqdm

import path_to_kapture
import kapture
import kapture.utils.logging
import kapture.io.csv as csv
import kapture.io.ply as ply
import kapture.io.image as image
import kapture.io.features
import kapture.io.records


logger = logging.getLogger('plot')


def plot_ply(kapture_path: str,
             ply_path: str,
             axis_length: float,
             only: [],
             skip: []
             ) -> None:
    """
    Plot the kapture data in a PLY file.

    :param kapture_path: top directory of the kapture
    :param ply_path: path to the ply file to create
    :param axis_length: length of axis representation (in world unit)
    :param only: list of the only kapture objects to plot (optional)
    :param skip: list of the kapture objects to skip
    """
    try:
        os.makedirs(ply_path, exist_ok=True)

        logger.info('loading data ...')
        kapture_data = csv.kapture_from_dir(kapture_path)

        def _should_do(choice: str) -> bool:
            return (not only or choice in only) and choice not in skip

        logger.info('plotting  ...')
        if _should_do('rigs') and kapture_data.rigs:
            logger.info(f'creating {len(kapture_data.rigs)} rigs.')
            for rig_id, rig in kapture_data.rigs.items():
                rig_ply_filepath = path.join(ply_path, f'rig_{rig_id}.ply')
                logger.info(f'creating rig file : {rig_ply_filepath}.')
                logger.debug(rig_ply_filepath)
                ply.rig_to_ply(rig_ply_filepath, rig, axis_length)

        if _should_do('trajectories') and kapture_data.trajectories:
            trajectories_ply_filepath = path.join(ply_path, 'trajectories.ply')
            logger.info(f'creating trajectories file : {trajectories_ply_filepath}')
            ply.trajectories_to_ply(trajectories_ply_filepath, kapture_data.trajectories, axis_length)

        if _should_do('points3d') and kapture_data.points3d:
            points3d_ply_filepath = path.join(ply_path, 'points3d.ply')
            logger.info(f'creating 3D points file : {points3d_ply_filepath}')
            ply.points3d_to_ply(points3d_ply_filepath, kapture_data.points3d)

        if _should_do('keypoints') and kapture_data.keypoints:
            logger.info(f'creating keypoints in 3D : {kapture.io.features.get_keypoints_fullpath(ply_path)}')
            keypoints_dsize = kapture_data.keypoints.dsize
            keypoints_dtype = kapture_data.keypoints.dtype
            keypoints_filepaths = kapture.io.features.keypoints_to_filepaths(kapture_data.keypoints, kapture_path)
            for image_filename, keypoints_filepath in tqdm(keypoints_filepaths.items(),
                                                           disable=logger.level >= logging.CRITICAL):
                image_filepath = kapture.io.records.get_image_fullpath(kapture_path, image_filename)
                image_keypoints_filepath = kapture.io.features.get_keypoints_fullpath(ply_path, image_filename) + '.jpg'
                image.image_keypoints_to_image_file(
                    image_keypoints_filepath, image_filepath, keypoints_filepath, keypoints_dtype, keypoints_dsize)

        logger.info('done.')

    except Exception as e:
        logging.critical(e)
        raise


def plot_command_line() -> None:
    """
    Do the plot to ply file using the parameters given on the command line.
    """
    plot_choices = {
        # cmd : help
        'rigs': 'plot the rig geometry, ie. relative pose of sensors into the rig.',
        'rig_stat': 'plot the sensor relative poses for each trajectory timestamp.',
        'trajectories': 'plot the trajectory of every sensors.',
        'points3d': 'plot the 3-D point cloud.',
        'keypoints': 'plot keypoints in 3D over the image plane.'
    }
    parser = argparse.ArgumentParser(description='plot out camera geometry to PLY file.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory')
    parser.add_argument('-o', '--output', required=False,
                        help='output directory (PLY file format).')
    parser.add_argument('--only', nargs='+', choices=plot_choices.keys(), default=[],
                        help='things to plot : ' + ' // '.join('{}: {}'.format(k, v) for k, v in plot_choices.items()))
    parser.add_argument('--skip', nargs='+', choices=plot_choices.keys(), default=['keypoints'],
                        help='things to not plot : ' + ' // '.join(plot_choices.keys()))
    parser.add_argument('--axis_length', type=float, default=0.1,
                        help='length of axis representation (in world unit).')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    if not args.output:
        args.output = args.input

    # only overwrite skip :
    if args.only:
        args.skip = []
    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v) for k, v in vars(args).items()]))
    plot_ply(args.input, args.output, args.axis_length, args.only, args.skip)


if __name__ == '__main__':
    plot_command_line()
