# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Functions to import the reconstruction part of colmap (only text format is supported).
"""

import logging
import numpy as np
import os.path as path
import re
from typing import Dict, Tuple, Optional
# kapture
import kapture
import kapture.io.features
# local
from .cameras import get_camera_kapture_id_from_colmap_id

logger = logging.getLogger('colmap')

colmap_reconstruction_split_pattern = r'[^,\s]+'


def import_from_colmap_cameras_txt(colmap_cameras_filepath: str) -> kapture.Sensors:
    """
    Imports Sensors from colmap cameras.txt

    :param colmap_cameras_filepath: input path to colmap cameras.txt file
    :return: kapture sensors
    """
    sensors = kapture.Sensors()
    # cameras[cam_id] = camera
    with open(colmap_cameras_filepath, 'r') as colmap_cameras_filepath:
        lines = colmap_cameras_filepath.readlines()
        # eliminate comments
        lines = (line for line in lines if not line.startswith('#'))
        # split by space and or comma
        lines = (re.findall(colmap_reconstruction_split_pattern, line.rstrip())
                 for line in lines)  # split fields
        for fields in lines:
            camera_id = get_camera_kapture_id_from_colmap_id(int(fields[0]))
            camera_type = str(fields[1])
            image_size = [str(s) for s in fields[2:4]]
            projection_params = [str(f) for f in fields[4:]]
            camera = kapture.Camera(camera_type, image_size + projection_params)
            sensors[camera_id] = camera
    return sensors


def import_from_colmap_images_txt(colmap_images_filepath: str,
                                  kapture_dirpath: Optional[str] = None
                                  ) -> Tuple[kapture.RecordsCamera, kapture.Trajectories, Optional[kapture.Keypoints]]:
    """
    Imports RecordsCamera, Trajectories and Keypoints from colmap images.txt

    :param colmap_images_filepath: path to colmap images.txt file
    :param kapture_dirpath: path to kapture root path.
                            If not given (None), keypoints are not created.
    :return: kapture images, trajectories and keypoints
    """

    # colmap images file format is :
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)

    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()
    keypoints = None
    image_names = []  # first pass
    # first pass: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    # to images and trajectories
    with open(colmap_images_filepath, 'r') as colmap_images_file:
        lines = colmap_images_file.readlines()
        lines = (line for line in lines if not line.startswith('#'))  # eliminate comments
        lines = (line for i, line in enumerate(lines) if (i % 2) == 0)  # eliminate even lines
        # split by space and or comma
        lines = (re.findall(colmap_reconstruction_split_pattern, line.rstrip())
                 for line in lines)  # split fields
        # but make sure not to split spaces in file names
        lines = (line[0:9] + [' '.join(line[9:])] for line in lines)
        for fields in lines:
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            timestamp = int(fields[0])  # use image ID as timestamp
            q = [float(v) for v in fields[1:5]]
            t = [float(v) for v in fields[5:8]]
            pose = kapture.PoseTransform(q, t)
            camera_id = get_camera_kapture_id_from_colmap_id(int(fields[8]))
            image_name = fields[9]
            images[timestamp, camera_id] = image_name
            trajectories[timestamp, camera_id] = pose
            image_names.append(image_name)

    # second pass: keypoints, observations and points 3d
    if kapture_dirpath is not None:
        # second pass: POINTS2D[] as (X, Y, POINT3D_ID)
        image_names_with_keypoints = set()
        # observations = kapture.Observations()
        with open(colmap_images_filepath, 'r') as colmap_images_file:
            lines = colmap_images_file.readlines()
            lines = (line for line in lines if not line.startswith('#'))  # eliminate comments
            lines = (line for i, line in enumerate(lines) if (i % 2) == 1)  # eliminate odd lines
            # split by space and or comma
            lines = (re.findall(colmap_reconstruction_split_pattern, line.rstrip())
                     for line in lines)  # split fields
            for image_name, fields in zip(image_names, lines):
                image_keypoints_colmap = np.array(fields).reshape((-1, 3))[:, 0:2].astype(np.float32)
                # register as keypoints if there is at least one
                if image_keypoints_colmap.shape[0] > 0:
                    keypoints_filepath = kapture.io.features.get_keypoints_fullpath(kapture_dirpath, image_name)
                    kapture.io.features.image_keypoints_to_file(keypoints_filepath, image_keypoints_colmap)
                    image_names_with_keypoints.add(image_name)
                    # TODO: observations

        if image_names_with_keypoints:
            keypoints = kapture.Keypoints('SIFT', np.float32, 2, image_names_with_keypoints)

    return images, trajectories, keypoints


def import_from_colmap_points3d_txt(colmap_points3d_filepath: str,
                                    image_names: Dict[int, str] = None,
                                    skip_observations: bool = False
                                    ) -> Tuple[kapture.Points3d, Optional[kapture.Observations]]:
    """
    Imports the colmap file named "points3d.txt" containing both points 3D and observations.

    :param colmap_points3d_filepath: path to the colmap file named "points3d.txt"
    :param image_names: dict of image names matching the colmap image id to the kapture image name
                        colmap_image_idx -> kapture_image_filename
    :param skip_observations: skip import of observations id true.
    :return: kapture 3D points and observations
    """
    assert path.basename(colmap_points3d_filepath) == 'points3D.txt'
    points3d = []
    observations = kapture.Observations()

    # colmap points3D.txt contains both points 3D and observations
    with open(colmap_points3d_filepath, 'r') as file:
        # in the reconstruction, spaces and commas can be used as separators
        lines = file.readlines()
        # eliminate comments
        lines = (line for line in lines if not line.startswith('#'))
        # split by space and or comma
        lines = ([float(value) for value in re.findall(colmap_reconstruction_split_pattern, values)]
                 for values in lines)  # split into an array of floats
        for index, values in enumerate(lines):
            points3d.append(values[1:4] + values[4:7])
            if not skip_observations and len(values) > 8 and len(image_names) > 0:
                for image_idx, point_2d_idx in zip(values[8::2], values[9::2]):
                    filename = image_names.get(int(image_idx), 'unknown')
                    observations.add(index, filename, int(point_2d_idx))

    points3d = kapture.Points3d(np.array(points3d)) if points3d else kapture.Points3d()
    observations = None if len(observations) == 0 else observations
    return points3d, observations
