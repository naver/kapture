#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a 4seasons model into a kapture.
The data structure is defined here:
    https://www.4seasons-dataset.com/documentation

The 4Seasons dataset contains recordings from a stereo-inertial camera system coupled with a high-end RTK-GNSS.

For each sequence, the recorded data is stored in the following structure:
├── KeyFrameData
├── distorted_images ...
├── undistorted_images
│   ├── cam0
│   └── cam1
├── GNSSPoses.txt
├── Transformations.txt
├── imu.txt
├── result.txt
├── septentrio.nmea
└── times.txt

The calibration folder has the following structure:
├── calib_0.txt
├── calib_1.txt
├── calib_stereo.txt
├── camchain.yaml
├── undistorted_calib_0.txt
├── undistorted_calib_1.txt
└── undistorted_calib_stereo.txt
"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from glob import glob
#from PIL import Image
from tqdm import tqdm
# kapture
import path_to_kapture  # noqa: F401
from kapture.core.Sensors import SENSOR_TYPE_DEPTH_CAM
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from typing import List, Dict

logger = logging.getLogger('4seasons')


def load_4seasons_sensors(
        calibration_dir_path: str
):
    # this dataset is made with a single stereo camera (2 cams).

    sensors = kapture.Sensors()
    """
    Pinhole 501.4757919305817 501.4757919305817 421.7953735163109 167.65799492501083 0.0 0.0 0.0 0.0
    800 400
    crop
    800 400
    """
    intrinsic_file_names = {
        'cam0': 'undistorted_calib_0.txt',
        'cam1': 'undistorted_calib_1.txt',
    }
    for cam_id, intrinsic_file_name in intrinsic_file_names.items():
        intrinsic_file_path = path.join(calibration_dir_path, intrinsic_file_name)
        with open(intrinsic_file_path, 'r') as intrinsic_file:
            line = intrinsic_file.readline().split(' ')
            # 1st line looks like:
            #     Pinhole 501.4757919305817 501.4757919305817 421.7953735163109 167.65799492501083 0.0 0.0 0.0 0.0
            # assuming fx fy cx cy distortion coef (null)
            if line[0] != 'Pinhole':  # sanity check
                raise ValueError(f'unexpected camera model {line[0]} (only is Pinhole valid).')
            fx, fy, cx, cy = (float(e) for e in line[1:5])

            line = intrinsic_file.readline().split(' ')
            # second line looks like :
            #   800 400
            # assuming image_width image_height
            w, h = (int(e) for e in line)
            sensors[cam_id] = kapture.Camera(kapture.CameraType.PINHOLE, [w, h, fx, fy, cx, cy], name=cam_id)

    # rigs
    rigs = kapture.Rigs()
    stereo_matrix_file_name = path.join(calibration_dir_path, "undistorted_calib_stereo.txt")
    # 4x4 matrix denoting the rigid transformation from XXX to XXX camera.
    # cam0 = left, cam1 = right
    pose_matrix = np.loadtxt(stereo_matrix_file_name)
    r = pose_matrix[0:3, 0:3]
    q = quaternion.from_rotation_matrix(r)
    t = pose_matrix[0:3, 3]
    cam0_from_cam1 = kapture.PoseTransform(q, t)
    rigs['rig', 'cam0'] = kapture.PoseTransform()
    # TODO

    # trajectories from
    return sensors, rigs


def load_times_ids(
        times_file_path: str
):
    table = np.loadtxt(times_file_path)
    shots_ids = table[:, 0].astype(int)
    timestamps_ns = (table[:, 1] * 1e9).astype(int)
    shot_id_to_timestamp = {}
    for shot_id, timestamp_ns in zip(shots_ids, timestamps_ns):
        timestamp_ns = int(timestamp_ns)
        shot_id = f'{shot_id:19d}'
        # shot_id should match timestamp in ns
        assert str(timestamp_ns) == shot_id
        shot_id_to_timestamp[shot_id] = timestamp_ns

    return shot_id_to_timestamp


def import_4seasons_images(
        recording_dir_path: str,
        kapture_dir_path: str,
        shot_id_to_timestamp: Dict[str, int],
        sensors: kapture.Sensors,
        images_import_method: TransferAction,
):
    kapture_images = kapture.RecordsCamera()
    logger.info('importing images ...')
    season_image_dir_path = path.join(recording_dir_path, 'undistorted_images')
    for sensor_id in sensors:
        for shot_id, timestamp_ns  in shot_id_to_timestamp.items():
            image_file_name = path.join(sensor_id, f'{shot_id}.png')
            kapture_images[timestamp_ns, sensor_id] = image_file_name

    filename_list = [f for _, _, f in kapture.flatten(kapture_images)]
    import_record_data_from_dir_auto(
        source_record_dirpath=season_image_dir_path,
        destination_kapture_dirpath=kapture_dir_path,
        filename_list=filename_list,
        copy_strategy=images_import_method)
    return kapture_images


def load_4season_keyframe_data(
        keyframes_file_path: str
):
    # first pass, read pose
    with open(keyframes_file_path, 'rt') as fin:
        for i in range(10):
            line = fin.readline()
            if line.startswith('# camToWorld:'):
                break
        if not line.startswith('# camToWorld:'):
            raise ValueError(f'pose not found in {path.basename(keyframes_file_path)}')
        # translation vector, rotation quaternion
        line = fin.readline()
    line = [float(v) for v in line.split(',')]
    t = line[0:3]
    q = line[3:]
    pose = kapture.PoseTransform(r=q, t=t)
    # pose found
    return pose


def load_4seasons_keyframes(
        keyframes_dir_path: str,
        kapture_dir_path: str,
        shot_id_to_timestamp: Dict[str, int],
        sensors: kapture.Sensors,
):
    trajectories = kapture.Trajectories()
    #                                   KeyFrame_1602074967051661568.txt
    keyframe_filename_re = re.compile(r'^KeyFrame_(?P<shot_id>\d{19})\.txt$')
    filename_it = (path.basename(f) for f in glob(path.join(keyframes_dir_path, '*.txt')))
    filename_it = (filename for filename in filename_it
                   if keyframe_filename_re.match(filename))
    filename_it = (keyframe_filename_re.match(filename) for filename in filename_it)
    filename_it = {filename[0]: filename[1] for filename in filename_it}
    for filename, shot_id in filename_it.items():
        assert shot_id in shot_id_to_timestamp
        timestamp_ns = shot_id_to_timestamp[shot_id]
        keyframes_file_path = path.join(keyframes_dir_path, filename)
        pose = load_4season_keyframe_data(keyframes_file_path).inverse()
        trajectories[timestamp_ns, 'rig'] = pose

    return trajectories


def _import_4seasons_sequence(
        calibration_dir_path: str,
        recording_dir_path: str,
        kapture_dir_path: str,
        images_import_method: TransferAction,
        force_overwrite_existing: bool):
    os.makedirs(kapture_dir_path, exist_ok=True)
    # delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    # snapshots = kapture.RecordsCamera()
    # depth_maps = kapture.RecordsDepth()
    # trajectories = kapture.Trajectories()
    # rigs = kapture.Rigs()
    # sensors = kapture.Sensors()

    """
    recording_dir_path contains : 
    KeyFrameData: contains the KeyFrameFiles.
    distorted_images: contains both the distorted images from the left and right camera, respectively.
    undistorted_images: contains both the undistorted images from the left and right camera, respectively.
    GNSSPoses.txt: is a list of 7DOF globally optimized poses (include scale from VIO to GNSS frame) for all keyframes (after GNSS fusion and loop closure detection). Each line is specified as frame_id, translation (t_x, t_y, t_z), rotation as quaternion (q_x, q_y, q_z, w), scale, fusion_quality (not relevant), and v3 (not relevant).
    Transformations.txt: defines transformations between different coordinate frames.
    imu.txt: contains raw IMU measurements. Each line is specified as frame_id, (angular velocity (w_x, w_y, w_z), and linear acceleration (a_x, a_y, a_z)).
    result.txt: contains the 6DOF visual interial odometry poses for every frame (not optimized). Each line is specified as timestamp (in seconds), translation (t_x, t_y, t_z), rotation as quaternion (q_x, q_y, q_z, w).
    septentrio.nmea: contains the raw GNSS measurements in the NMEA format.
    times.txt is a list of times in unix timestamps (in seconds), and exposure times (in milliseconds) for each frame (frame_id, timestamp, exposure).
    """

    # sensors
    sensors, rigs = load_4seasons_sensors(calibration_dir_path=calibration_dir_path)

    # timestamps
    times_filename = path.join(recording_dir_path, 'times.txt')
    shot_id_to_timestamp = load_times_ids(times_filename)

    # images
    if False:
        kapture_images = import_4seasons_images(
            recording_dir_path=recording_dir_path,
            kapture_dir_path=kapture_dir_path,
            shot_id_to_timestamp=shot_id_to_timestamp,
            sensors=sensors,
            images_import_method=images_import_method
        )

    # trajectorie uses keyframes
    keyframes_dir_path = path.join(recording_dir_path, 'KeyFrameData')
    trajectories = load_4seasons_keyframes(
        keyframes_dir_path=keyframes_dir_path,
        kapture_dir_path=kapture_dir_path,
        shot_id_to_timestamp=shot_id_to_timestamp,
        sensors=sensors,
    )

    # imu.txt defines accel and gyro
    imported_kapture = kapture.Kapture(
        sensors=sensors,
        rigs=rigs,
        trajectories=trajectories)
    kapture_to_dir(kapture_dir_path, imported_kapture)


def import_4seasons(
        d4seasons_dir_path: str,
        kapture_dir_path: str,
        images_import_method: TransferAction = TransferAction.skip,
        force_overwrite_existing: bool = False
) -> None:
    """
    Imports 4seasons dataset and save them as kapture.
    :param d4seasons_dir_path: path to the 4seasons root path
    :param kapture_dir_path: path to kapture output top directory
    :param images_import_method: choose how to import actual image files
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    """
    os.makedirs(kapture_dir_path, exist_ok=True)

    # sensors
    calibration_dir_path = path.join(d4seasons_dir_path, 'calibration')

    # recordings
    recording_names = [
        'recording_2020-10-07_14-47-51'
    ]
    recording_dir_paths = (path.join(d4seasons_dir_path, recording_name) for recording_name in recording_names)
    for recording_dir_path in recording_dir_paths:
        logger.debug(f'processing : {recording_dir_path}')
        _import_4seasons_sequence(
            calibration_dir_path=calibration_dir_path,
            recording_dir_path=recording_dir_path,
            kapture_dir_path=kapture_dir_path,
            images_import_method=images_import_method,
            force_overwrite_existing=force_overwrite_existing
        )


def import_4seasons_command_line() -> None:
    """
    Imports 4seasons dataset and save them as kapture using the parameters given on the command line.
    It assumes images are undistorted.
    """
    parser = argparse.ArgumentParser(
        description='Imports 4seasons dataset files to the kapture format.')
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
    parser.add_argument('-i', '--input', required=True, help='input path to 4seasons root path')
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

    import_4seasons(d4seasons_dir_path=args.input,
                    kapture_dir_path=args.output,
                    images_import_method=args.image_transfer,
                    force_overwrite_existing=args.force)


if __name__ == '__main__':
    import_4seasons_command_line()
