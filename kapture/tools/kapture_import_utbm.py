#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a UTBM dataset stored as rosbags: one with images from several camera,
and one with GPS position associated to the images recorded.

This is an outdoor dataset recorded with a car in various conditions.
More info at:
https://epan-utbm.github.io/utbm_robocar_dataset/
"""

import logging
import sys
from typing import List
# kapture
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

# rosbag
try:
    import rosbag  # noqa: F401
    from kapture.converter.ros_tools.import_rosbag import RosBagImporter
    has_rosbag = True
except ModuleNotFoundError:
    has_rosbag = False

from kapture.converter.ros_tools.import_utbm_sensor import BB2_CAMERA_IDENTIFIERS, TOPICS_BB2
from kapture.converter.ros_tools.import_utbm_sensor import XB3_CAMERA_IDENTIFIERS, TOPICS_XB3
from kapture.converter.ros_tools.import_utbm_sensor import import_utbm_sensors

logger = logging.getLogger('rosbag')  # Using global logger


def import_utbm_rosbag(camera_info: List[str],
                       kapture_output: str,
                       bag_file: str = 'input.bag',
                       image_topics: List[str] = TOPICS_BB2,
                       camera_identifiers: List[str] = BB2_CAMERA_IDENTIFIERS,
                       force_same_time: bool = True,
                       bb2: bool = False,
                       xb3: bool = False,
                       percent: int = 100,
                       force: bool = False,
                       verbose: Verbosity = DEFAULT_VERBOSITY,
                       quiet: bool = False) -> None:
    """
    Imports UTBM rosbags to kapture format.

    :param camera_info: list of cameras calibration yaml files as published on UTBM github
    :param kapture_output: directory where to save kapture files.
    :param bag_file: bag file path
    :param image_topics: list of image topics to import, defaults to BB2 values
    :param camera_identifiers: list of camera identifiers as defined in the sensors file of the kapture rig
                               and corresponding to the image topics in the same order; defaults to BB2 values
    :param force_same_time: force that images are taken at the same time
    :param bb2: set BB2 image topics and camera identifiers
    :param xb3: set XB3 image topics and camera identifiers
    :param percent: percentage of images to keep
    :param force: silently delete kapture data if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    # Change the default logging
    root_logger = logging.getLogger()
    root_logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s::%(name)s: %(message)s'))
    logger.addHandler(ch)
    setup_logging(logger, verbose, quiet)
    if not has_rosbag:
        logger.fatal("You don't have ROS or import_rosbag installed")
        sys.exit(-1)
    # Read camera calibration files to make a sensors object
    sensors = import_utbm_sensors(camera_info)
    importer = RosBagImporter(bag_file, None, sensors, kapture_output, force)
    if bb2:
        image_topics = TOPICS_BB2
        camera_identifiers = BB2_CAMERA_IDENTIFIERS
    elif xb3:
        image_topics = TOPICS_XB3
        camera_identifiers = XB3_CAMERA_IDENTIFIERS
    else:
        if not image_topics:
            image_topics = TOPICS_BB2
        if not camera_identifiers:
            camera_identifiers = BB2_CAMERA_IDENTIFIERS
    importer.import_multi_camera(None, image_topics, camera_identifiers, False, False, percent)
    importer.save_to_kapture()


def main():
    try:
        run_cli(import_utbm_rosbag, aliases={
            '--bag_file': ['-b'],
            '--image_topics': ['-i'],
            '--camera_identifiers': ['-c'],
            '--percent': ['-p'],
            '--kapture_output': ['-k'],
        })
    except (AssertionError, OSError, ValueError) as any_ex:
        logger.fatal(f'Fatal error: {any_ex}')


if __name__ == '__main__':
    main()
