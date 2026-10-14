#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a rosbag with images from several camera and position associated to the images recorded.
Works by default with the RealSense T265 camera.
"""

import logging
from os import path
import sys
from typing import List, Optional
# kapture
import kapture.io.csv as kcsv
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

# rosbag
try:
    import rosbag  # noqa: F401
    from kapture.converter.ros_tools.import_rosbag import RosBagImporter

    has_rosbag = True
except ModuleNotFoundError:
    has_rosbag = False

logger = logging.getLogger('rosbag')  # Using global logger

# Default ROS topics: RealSense T265 camera
TOPIC_ODOMETRY = '/camera/odom/sample'
TOPIC_CAMERA_LEFT = '/camera/fisheye1/image_raw'
TOPIC_CAMERA_RIGHT = '/camera/fisheye2/image_raw'
DEFAULT_IMAGE_TOPICS = [TOPIC_CAMERA_LEFT, TOPIC_CAMERA_RIGHT]


def import_rosbag(kapture_rig: str,
                  kapture_output: str,
                  bag_file: str = 'input.bag',
                  odometry_topic: Optional[str] = TOPIC_ODOMETRY,
                  image_topics: List[str] = DEFAULT_IMAGE_TOPICS,
                  camera_identifiers: List[str] = [],
                  all_pose: bool = False,
                  match_pose: bool = True,
                  percent: int = 100,
                  force: bool = False,
                  verbose: Verbosity = DEFAULT_VERBOSITY,
                  quiet: bool = False) -> None:
    """
    Imports ROS bag file to kapture format.

    :param kapture_rig: kapture with rig and sensors definitions for the capturing equipment
                        corresponding to the ROS bag
    :param kapture_output: directory where to save kapture files.
    :param bag_file: bag file path
    :param odometry_topic: odometry topic to import
    :param image_topics: list of image topics to import
    :param camera_identifiers: list of camera identifiers as defined in the sensors file of the kapture rig
                               and corresponding to the image topics in the same order
    :param all_pose: save all poses from the odometry topic to the trajectory
    :param match_pose: find a matching pose for every image, and save it in the trajectory with the same timestamp
    :param percent: percentage of images to keep
    :param force: silently delete kapture data if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    if not has_rosbag:
        logger.fatal("You don't have ROS or import_rosbag installed")
        sys.exit(-1)
    if not camera_identifiers:
        raise ValueError('camera_identifiers is required')
    # Import predefined rig and sensors list
    rig_sensors_kapture_path = path.abspath(kapture_rig)
    kapture_data = kcsv.kapture_from_dir(rig_sensors_kapture_path)
    rigs = kapture_data.rigs
    rig_id = None
    if odometry_topic is not None:
        # If we have trajectory points, we need a rig with a unique id
        if kapture_data.rigs is None or len(kapture_data.rigs) == 0:
            raise ValueError(f'Rig definition is empty in {rig_sensors_kapture_path}')
        for r_id in rigs.keys():
            if rig_id is None:
                rig_id = r_id
            elif r_id != rig_id:
                raise ValueError(f'Found rig_id {r_id} and {rig_id}:'
                                 f' there should be only one rig_id defined in {rig_sensors_kapture_path}')
        if rig_id is None:
            raise ValueError(f'No rig defined in {kapture_data}')
    if kapture_data.cameras is None or len(kapture_data.cameras) == 0:
        raise ValueError(f'Cameras definition is empty in {rig_sensors_kapture_path}')

    importer = RosBagImporter(bag_file, rigs, kapture_data.sensors, kapture_output, force)
    # Import the images
    if not image_topics:
        image_topics = DEFAULT_IMAGE_TOPICS
    importer.import_multi_camera(odometry_topic, image_topics, camera_identifiers, True,
                                 all_pose, match_pose, percent)
    importer.save_to_kapture(rig_id)


def main():
    try:
        run_cli(import_rosbag, aliases={
            '--bag_file': ['-b'],
            '--kapture_rig': ['-r'],
            '--odometry_topic': ['-o'],
            '--image_topics': ['-i'],
            '--camera_identifiers': ['-c'],
            '--all_pose': ['-a'],
            '--match_pose': ['-m'],
            '--percent': ['-p'],
            '--kapture_output': ['-k'],
        })
    except (AssertionError, OSError, ValueError) as any_ex:
        logger.fatal(f'Fatal error: {any_ex}')


if __name__ == '__main__':
    main()
