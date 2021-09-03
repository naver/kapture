# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Import all types of UTBM sensors information files to make the sensors list.
These files are published on github at:
https://github.com/epan-utbm/utbm_robocar_dataset/tree/baselines/camera_info

The files have been produced by the ROS camera_calibration package:
https://wiki.ros.org/camera_calibration
"""

from typing import List, Union

from kapture.core.Sensors import Sensors
from kapture.utils.open_cv import import_opencv_camera_calibration

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
        sensor = import_opencv_camera_calibration(camera_info_file)
        sensors[sensor.name] = sensor
    return sensors
