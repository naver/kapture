#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a UTBM dataset stored as rosbags: one with images from several camera,
and one with GPS position associated to the images recorded.

This is an outdoor dataset recorded with a car in various conditions.
More info at:
https://epan-utbm.github.io/utbm_robocar_dataset/
"""

import argparse
import logging
import sys
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture.core.Sensors
import kapture.utils.logging

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


def import_utbm_rosbag(args) -> None:
    """
    Import the UTBM rosbag with the parameters.

    :param args: arguments to use
    """
    # Read camera calibration files to make a sensors object
    sensors = import_utbm_sensors(args.camera_info)
    importer = RosBagImporter(args.bag_file, None, sensors, args.kapture_output, args.force)
    if args.bb2:
        image_topics = TOPICS_BB2
        camera_identifiers = BB2_CAMERA_IDENTIFIERS
    elif args.xb3:
        image_topics = TOPICS_XB3
        camera_identifiers = XB3_CAMERA_IDENTIFIERS
    else:
        image_topics = args.image_topics
        if not image_topics:
            image_topics = TOPICS_BB2
        camera_identifiers = args.camera_identifiers
        if not camera_identifiers:
            camera_identifiers = BB2_CAMERA_IDENTIFIERS
    importer.import_multi_camera(None, image_topics, camera_identifiers, False, False, args.percent)
    importer.save_to_kapture()


def import_utbm_rosbag_command_line() -> None:
    """
    Do the UTBM to kapture import using the command line parameters provided by the user.
    """
    parser = argparse.ArgumentParser(description='Imports UTBM rosbags to kapture format.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet', action='store_const', dest='verbose',
                                  const=logging.CRITICAL)
    parser.add_argument('-y', '--force', action='store_true', default=False,
                        help='silently delete kapture data if already exists.')
    # create the parser for the import command #########################################################################
    parser.add_argument('-b', '--bag_file', default='input.bag',
                        help='bag file path', required=False)
    parser.add_argument('--camera_info', nargs="*",
                        help='list of cameras calibration yaml files as published on UTBM github', required=True)
    parser.add_argument('-i', '--image_topics', nargs="*",
                        help=f'list of image topics to import, defaults to BB2 values {TOPICS_BB2}',
                        required=False)
    parser.add_argument('-c', '--camera_identifiers', nargs="*",
                        help=f'list of camera identifiers as defined in the sensors file of the kapture rig'
                             ' and corresponding to the image topics in the same order;'
                             f' defaults to BB2 values {BB2_CAMERA_IDENTIFIERS}',
                        required=False)
    parser.add_argument('--bb2', action='store_true', default=False,
                        help='Set BB2 image topics and camera identifiers')
    parser.add_argument('--xb3', action='store_true', default=False,
                        help='Set XB3 image topics and camera identifiers')
    parser.add_argument('-p', '--percent', type=int, default=100,
                        help='percentage of images to keep', required=False)
    parser.add_argument('-k', '--kapture_output', required=True,
                        help='directory where to save Kapture files.')
    ####################################################################################################################
    args = parser.parse_args()

    # Change the default logging
    root_logger = logging.getLogger()
    root_logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s::%(name)s: %(message)s'))
    logger.addHandler(ch)
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # for debug, let kapture express itself.
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    logger.debug(f'{sys.argv[0]} \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()))

    import_utbm_rosbag(args)


if __name__ == '__main__':
    if not has_rosbag:
        logger.fatal("You don't have ROS or import_rosbag installed")
        sys.exit(-1)
    try:
        import_utbm_rosbag_command_line()
    except (AssertionError, OSError, ValueError) as any_ex:
        logger.fatal(f'Fatal error: {any_ex}')
