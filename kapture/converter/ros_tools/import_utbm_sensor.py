# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Import all types of UTBM sensors information files to make the sensors list.
These files are published on github at:
https://github.com/epan-utbm/utbm_robocar_dataset/tree/baselines/camera_info

The files have been produced by the ROS camera_calibration package:
https://wiki.ros.org/camera_calibration
"""

from os import path
from typing import List, Union
import yaml

from kapture.core.Sensors import CameraType, Camera, Sensors

#  UTBM specific constants values

# Bumblebee2 cameras ROS topics and identifiers in same order:
TOPICS_BB2 = ['/BB2_9211442/left/image_raw',
              '/BB2_9211442/right/image_raw']
BB2_CAMERA_IDENTIFIERS = ['bb2_left', 'bb2_right']

# Bumblebee XB3 cameras ROS topics and identifiers in same order:
TOPICS_XB3 = ['/BBX3_7140017/camera/center/image_raw',
              '/BBX3_7140017/camera/left/image_raw',
              '/BBX3_7140017/camera/right/image_raw']
XB3_CAMERA_IDENTIFIERS = ['bb_xb3_center', 'bb_xb3_left', 'bb_xb3_right']


def import_ros_camera_calibration(camera_info_file_path: str) -> Camera:
    """
    Parse a ROS camera calibration file to insert in the sensor list.

    :param camera_info_file_path: path to the camera info file
    :return: the camera name and its parameters
    """
    camera_info_file_path = path.abspath(camera_info_file_path)
    with open(camera_info_file_path) as file:
        camera_info = yaml.full_load(file)
        camera_name = camera_info['camera_name']
        w, h = camera_info['image_width'], camera_info['image_height']
        camera_matrix = camera_info['camera_matrix']['data']
        fx, cx, fy, cy = camera_matrix[0], camera_matrix[2], camera_matrix[4], camera_matrix[5]
        # Will use opencv model with parameters: w, h, fx, fy, cx, cy, k1, k2, p1, p2
        distortion_coefficients = camera_info['distortion_coefficients']['data']
        # Make sure k3 is null
        assert(distortion_coefficients[4] == float(0.0))
        camera_parameters = [w, h, fx, fy, cx, cy] + distortion_coefficients[0:4]
        camera = Camera(CameraType.OPENCV, camera_parameters, camera_name)
        return camera


def import_utbm_sensors(camera_info_file_paths: Union[str, List[str]]) -> Sensors:
    """
    Import all camera info.

    :param camera_info_file_paths: the list of camera info file paths
    :return: kapture Sensors with all camera defined
    """
    sensors = Sensors()
    if isinstance(camera_info_file_paths, str):
        camera_info_file_paths = [camera_info_file_paths]
    for camera_info_file in camera_info_file_paths:
        sensor = import_ros_camera_calibration(camera_info_file)
        sensors[sensor.name] = sensor
    return sensors
