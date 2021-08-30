# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
OpenCV related helper operations.
"""

import os.path as path
import yaml

from kapture.core.Sensors import CameraType, Camera


def import_opencv_camera_calibration(camera_info_file_path: str) -> Camera:
    """
    Parse an OpenCV camera calibration file to insert in the sensor list.

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
