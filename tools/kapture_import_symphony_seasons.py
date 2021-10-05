#!/usr/bin/env python3
# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import os.path as path
import quaternion
from typing import Optional

import path_to_kapture  # noqa: F401
import kapture
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.utils.logging
from kapture.io.records import TransferAction, import_record_data_from_dir_auto


logger = logging.getLogger('import_symphony_seasons')
MAPPING_FILES = ['db.txt']
QUERY_FILES = ['autumn-dawn.txt', 'autumn-overcast.txt', 'spring-dawn.txt', 'spring-sun.txt', 'winter-overcast.txt',
               'autumn-dusk.txt', 'autumn-sun.txt', 'spring-foggy.txt', 'winter-dusk.txt', 'winter-sun.txt',
               'autumn-foggy.txt', 'spring-overcast.txt', 'winter-foggy.txt']

SENSOR = kapture.Camera(kapture.CameraType.OPENCV, [704.0, 480.0,
                                                    780.170806, 709.378535,
                                                    317.745657, 246.801583,
                                                    -0.295703, 0.157403, -0.001469, -0.000924
                                                    ])
SENSOR_ID = 'camera_0'


def import_symphony_seasons(input_path: str,
                            kapture_path: str,
                            force_overwrite_existing: bool = False,
                            images_import_method: TransferAction = TransferAction.skip,
                            partition: Optional[str] = None):
    """
    Imports symphony_seasons dataset and save them as kapture.
    :param input_path: path to the symphony_seasons root path
    :param kapture_path: path to kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    :param partition: if specified = 'mapping' or 'query'
    """
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    if partition is None:
        listfiles = MAPPING_FILES + QUERY_FILES
    elif partition == 'mapping':
        listfiles = MAPPING_FILES
    elif partition == 'query':
        listfiles = QUERY_FILES
    else:
        raise ValueError(f'Unknown partition {partition}')

    cameras = kapture.Sensors()
    cameras[SENSOR_ID] = SENSOR
    snapshots = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()

    timestamp = 0
    image_added = set()
    for listfile in listfiles:
        listfile_fullpath = path.join(input_path, 'file_list', listfile)
        with open(listfile_fullpath, 'r') as file:
            lines_splits = [line.rstrip("\n\r").split() for line in file.readlines() if line.strip()]
        for line_split in lines_splits:
            filename = line_split[0]
            if filename in image_added:
                # duplicate
                continue
            snapshots[timestamp, SENSOR_ID] = filename
            image_added.add(filename)
            if len(line_split) > 1:
                rotation = [float(line_split[v]) for v in range(1, 5)]
                rotation = quaternion.from_float_array(rotation)
                camera_position = [float(line_split[v]) for v in range(5, 8)]
                trajectories[timestamp, SENSOR_ID] = kapture.PoseTransform(
                    rotation.inverse(), camera_position).inverse()
            timestamp += 1
    logger.info('copying image files ...')
    image_filenames = [f for _, _, f in kapture.flatten(snapshots)]
    import_record_data_from_dir_auto(input_path, kapture_path, image_filenames, images_import_method)

    # Save kapture data
    kapture_data = kapture.Kapture(sensors=cameras,
                                   trajectories=trajectories or None,
                                   records_camera=snapshots)
    logger.info(f'Saving to kapture {kapture_path}')
    kapture_to_dir(kapture_path, kapture_data)


def import_symphony_seasons_command_line():
    """
    Import symphony_seasons command line
    """
    parser = argparse.ArgumentParser(
        description='Imports symphony_seasons files to the kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i', '--input', required=True,
                        help='input path symphony_seasons root path')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    parser.add_argument('-p', '--partition', default=None, choices=['mapping', 'query'],
                        help='limit to mapping or query sequences only (using authors split files).')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_symphony_seasons(args.input, args.output, args.force,
                            args.image_transfer, args.partition)


if __name__ == '__main__':
    import_symphony_seasons_command_line()
