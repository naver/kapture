# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
OpenCV related helper operations.
"""

import os.path as path
import pathlib
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
        # First check a glitch coming from OpenCV 3: the YAML file produced is not compliant with PYYAML
        first_line = file.readline()
        if first_line.startswith('%YAML:'):
            first_line = ''
        yaml_content = first_line
        for line in file.readlines():
            # Check if they are some unsupported !!
            start_invalid = line.find('!!')
            if start_invalid != -1:
                # Remove everything after the !! except the end of line character
                line = line[0:start_invalid] + line[-1:]
            yaml_content = yaml_content + line
        camera_info = yaml.full_load(yaml_content)
        if 'camera_name' in camera_info:
            camera_name = camera_info['camera_name']
        else:
            # Comes straight from OpenCV without name: use file name instead
            camera_name = pathlib.Path(path.basename(camera_info_file_path)).stem
        w, h = camera_info['image_width'], camera_info['image_height']
        camera_matrix = camera_info['camera_matrix']['data']
        fx, cx, fy, cy = camera_matrix[0], camera_matrix[2], camera_matrix[4], camera_matrix[5]
        # Will use opencv model with parameters: w, h, fx, fy, cx, cy, k1, k2, p1, p2
        distortion_coefficients = camera_info['distortion_coefficients']['data']
        camera_type = CameraType.OPENCV
        if distortion_coefficients[4] == float(0.0):
            # No k3: opencv model with k1 k2 p1 p2
            camera_parameters = [w, h, fx, fy, cx, cy] + distortion_coefficients[0:4]
        else:
            camera_type = CameraType.FULL_OPENCV
            # must provide k1, k2, p1, p2, k3, k4, k5, k6
            # k3 is here but not necessarily k4, k5 and k6
            k4 = distortion_coefficients[5] if len(distortion_coefficients) > 5 else float(0)
            k5 = distortion_coefficients[6] if len(distortion_coefficients) > 6 else float(0)
            k6 = distortion_coefficients[7] if len(distortion_coefficients) > 7 else float(0)
            camera_parameters = [w, h, fx, fy, cx, cy] + distortion_coefficients[0:5] + [k4, k5, k6]
        camera = Camera(camera_type, camera_parameters, camera_name)
        return camera
