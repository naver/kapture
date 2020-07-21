#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a 7scenes model into a kapture.
The RGB-D Dataset 7-Scenes data structure is defined here:
    https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

    Each sequence (seq-XX.zip) consists of 500-1000 frames. Each frame consists of three files:
     - Color: frame-XXXXXX.color.png (RGB, 24-bit, PNG)
     - Depth: frame-XXXXXX.depth.png (depth in millimeters, 16-bit, PNG, invalid depth is set to 65535).
     - Pose: frame-XXXXXX.pose.txt (camera-to-world, 4×4 matrix in homogeneous coordinates).

"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from PIL import Image
# kapture
import path_to_kapture
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('7scenes')
MODEL = kapture.CameraType.RADIAL


def import_7scenes(d7scenes_path: str,
                   kapture_dir_path: str,
                   force_overwrite_existing: bool = False,
                   images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Imports RGB-D Dataset 7-Scenes dataset and save them as kapture.

    :param d7scenes_path: path to the 7scenes sequence root path
    :param kapture_dir_path: path to kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    logger.info('loading all content ...')
    POSE_SUFFIX = 'pose'
    RGB_SUFFIX = 'color'
    DEPTH_SUFFIX = 'depth'
    CAMERA_ID = 'kinect'
    d7s_filename_re = re.compile(r'frame-(?P<timestamp>\d{6})\.(?P<suffix>\w*)\.(?P<ext>\w*)')

    # populate
    d7s_filenames = (path.basename(path.join(dp, fn))
                     for dp, _, fs in os.walk(d7scenes_path) for fn in fs)
    d7s_filenames = {filename: d7s_filename_re.match(filename).groupdict()
                     for filename in d7s_filenames
                     if d7s_filename_re.match(filename)}
    # d7s_filenames -> timestamp, suffix, ext
    if not d7s_filenames:
        raise ValueError('no pose file found: make sure the path to 7scenes sequence is valid.')

    # images
    logger.info('populating image files ...')
    d7s_filenames_images = ((int(v['timestamp']), filename)
                            for filename, v in d7s_filenames.items()
                            if v['suffix'] == RGB_SUFFIX)
    snapshots = kapture.RecordsCamera()
    for timestamp, image_filename in sorted(d7s_filenames_images):
        snapshots[timestamp, CAMERA_ID] = image_filename

    # poses
    logger.info('import poses files ...')
    d7s_filenames_poses = ((int(v['timestamp']), filename)
                           for filename, v in d7s_filenames.items()
                           if v['suffix'] == POSE_SUFFIX)
    trajectories = kapture.Trajectories()
    for timestamp, pose_filename in d7s_filenames_poses:
        pose_filepath = path.join(d7scenes_path, pose_filename)
        pose_mat = np.loadtxt(pose_filepath)  # camera-to-world, 4×4 matrix in homogeneous coordinates
        rotation_mat = pose_mat[0:3, 0:3]
        position_vec = pose_mat[0:3, 3]
        rotation_quat = quaternion.from_rotation_matrix(rotation_mat)
        pose_world_from_cam = kapture.PoseTransform(r=rotation_quat, t=position_vec)
        pose_cam_from_world = pose_world_from_cam.inverse()
        trajectories[timestamp, CAMERA_ID] = pose_cam_from_world

    # sensors
    """
    From authors: The RGB and depth camera have not been calibrated and we can’t provide calibration parameters at the 
    moment. The recorded frames correspond to the raw, uncalibrated camera images. In the KinectFusion pipeline we used 
    the following default intrinsics for the depth camera: Principle point (320,240), Focal length (585,585).
    """
    sensors = kapture.Sensors()
    sensors[CAMERA_ID] = kapture.Camera(
        name='kinect',
        camera_type=kapture.CameraType.SIMPLE_PINHOLE,
        camera_params=[640, 480, 585, 320, 240]  # w, h, f, cx, cy
    )

    # import (copy) image files.
    logger.info('copying image files ...')
    image_filenames = [f for _, _, f in kapture.flatten(snapshots)]
    import_record_data_from_dir_auto(d7scenes_path, kapture_dir_path, image_filenames, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(
        records_camera=snapshots,
        trajectories=trajectories,
        sensors=sensors)

    logger.info('writing imported data ...')
    kapture_to_dir(kapture_dir_path, imported_kapture)


def import_7scenes_command_line() -> None:
    """
    Imports RGB-D Dataset 7-Scenes dataset and save them as kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='Imports RGB-D Dataset 7-Scenes files to the kapture format.')
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
                        help='input path Dataset 7-Scenes sequence root path')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_7scenes(d7scenes_path=args.input,
                   kapture_dir_path=args.output,
                   force_overwrite_existing=args.force,
                   images_import_method=args.image_transfer)


if __name__ == '__main__':
    import_7scenes_command_line()
