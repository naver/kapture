#!/usr/bin/env python3
# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

import logging
import os
import os.path as path
import quaternion
from typing import Literal, Optional

import kapture
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY
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


def import_symphony_seasons(input: str,
                            output: str,
                            force: bool = False,
                            image_transfer: TransferAction = TransferAction.link_absolute,
                            partition: Optional[Literal['mapping', 'query']] = None,
                            verbose: Verbosity = DEFAULT_VERBOSITY,
                            quiet: bool = False):
    """
    Imports symphony_seasons dataset and save them as kapture.

    :param input: path to the symphony_seasons root path
    :param output: path to kapture top directory
    :param force: silently overwrite kapture files if already exists.
    :param image_transfer: choose how to import actual image files.
    :param partition: limit to mapping or query sequences only (using authors split files).
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)

    os.makedirs(output, exist_ok=True)
    delete_existing_kapture_files(output, force_erase=force)

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
        listfile_fullpath = path.join(input, 'file_list', listfile)
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
    import_record_data_from_dir_auto(input, output, image_filenames, image_transfer)

    # Save kapture data
    kapture_data = kapture.Kapture(sensors=cameras,
                                   trajectories=trajectories or None,
                                   records_camera=snapshots)
    logger.info(f'Saving to kapture {output}')
    kapture_to_dir(output, kapture_data)


def main():
    run_cli(import_symphony_seasons, aliases={'--partition': ['-p']})


if __name__ == '__main__':
    main()
