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
from typing import Optional
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

POSE_SUFFIX = 'pose'
RGB_SUFFIX = 'color'
DEPTH_SUFFIX = 'depth'
CAMERA_ID = 'kinect'
PARTITION_FILENAMES = {
    'mapping': 'TrainSplit.txt',
    'query': 'TestSplit.txt'
}


def import_7scenes(d7scenes_path: str,
                   kapture_dir_path: str,
                   force_overwrite_existing: bool = False,
                   images_import_method: TransferAction = TransferAction.skip,
                   partition: Optional[str] = None
                   ) -> None:
    """
    Imports RGB-D Dataset 7-Scenes dataset and save them as kapture.

    :param d7scenes_path: path to the 7scenes sequence root path
    :param kapture_dir_path: path to kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    :param partition: if specified = 'mapping' or 'query'. Requires d7scenes_path/TestSplit.txt or TrainSplit.txt
                    to exists.
    """
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    logger.info('loading all content ...')

    d7s_filename_re = re.compile(r'((?P<sequence>.+)/)?frame-(?P<frame_id>\d{6})\.(?P<suffix>\w*)\.(?P<ext>\w*)')

    # populate all relevant files
    d7s_filenames = (path.relpath(path.join(dp, fn), d7scenes_path)
                     for dp, _, fs in os.walk(d7scenes_path) for fn in fs)

    logger.info('populating 7-scenes files ...')
    d7s_filenames = {filename: d7s_filename_re.search(filename).groupdict()
                     for filename in sorted(d7s_filenames)
                     if d7s_filename_re.search(filename)}

    # reorg as shot[seq, id] = {color: , pose: , ...}
    shots = {}
    for timestamp, (filename, file_attribs) in enumerate(d7s_filenames.items()):
        shot_id = (file_attribs.get('sequence'), file_attribs['frame_id'])
        shots.setdefault(shot_id, {})[file_attribs['suffix']] = filename

    # fake timestamps
    for timestamp, shot_id in enumerate(shots):
        shots[shot_id]['timestamp'] = timestamp

    # if given, filter partition
    if partition is not None:
        # read the authors split file
        partition_filepath = path.join(d7scenes_path, PARTITION_FILENAMES[partition])
        if not path.isfile(partition_filepath):
            raise FileNotFoundError(f'partition file is missing: {partition_filepath}.')
        with open(partition_filepath, 'rt') as file:
            split_sequences = [f'seq-{int(seq.strip()[len("sequence"):]):02}' for seq in file.readlines()]
        assert len(split_sequences) > 0
        # filter out
        shots = {(seq, frame): shot
                 for (seq, frame), shot in shots.items()
                 if seq in split_sequences}

    if len(shots) == 0:
        raise FileNotFoundError('no file found: make sure the path to 7scenes sequence is valid.')

    # eg. shots['seq-01', '000000'] =
    #       {'color': 'seq-01/frame-000000.color.jpg', 'pose': 'seq-01/frame-000000.pose.txt', 'timestamp': 0}

    # images
    logger.info('populating image files ...')
    snapshots = kapture.RecordsCamera()
    for shot in shots.values():
        snapshots[shot['timestamp'], CAMERA_ID] = shot['color']

    # poses
    logger.info('import poses files ...')
    trajectories = kapture.Trajectories()
    for shot in shots.values():
        pose_filepath = path.join(d7scenes_path, shot['pose'])
        pose_mat = np.loadtxt(pose_filepath)  # camera-to-world, 4×4 matrix in homogeneous coordinates
        rotation_mat = pose_mat[0:3, 0:3]
        position_vec = pose_mat[0:3, 3]
        rotation_quat = quaternion.from_rotation_matrix(rotation_mat)
        pose_world_from_cam = kapture.PoseTransform(r=rotation_quat, t=position_vec)
        pose_cam_from_world = pose_world_from_cam.inverse()
        trajectories[shot['timestamp'], CAMERA_ID] = pose_cam_from_world

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
    parser.add_argument('-p', '--partition', default=None, choices=['mapping', 'query'],
                        help='limit to mapping or query sequences only (using authors split files).')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_7scenes(d7scenes_path=args.input,
                   kapture_dir_path=args.output,
                   force_overwrite_existing=args.force,
                   images_import_method=args.image_transfer,
                   partition=args.partition)


if __name__ == '__main__':
    import_7scenes_command_line()
