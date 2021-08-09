#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a RIO10 model into a kapture.
The RGB-D Dataset RIO10 data structure is defined here:
    https://github.com/WaldJohannaU/RIO10#dataformat

Sequence seq<scene_id>_01 is always the training sequence (with *.color.jpg, *.pose.txt and *.rendered.depth.png)
seqXX_02 is the validation sequence (with *.color.jpg, *.pose.txt and *.rendered.depth.png).
Please note that we do not provide the ground truth (including semantics) for our hidden test set
all seqXX_02+ are hidden sequences (only *.color.jpg and *.rendered.depth.png)
"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from PIL import Image
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
from typing import List

logger = logging.getLogger('RIO10')

TIMESTAMP_SEQUENCE_BREAK = 999
DEPTH_TO_METER = 1.0e-3  # mm to meter

MAPPING_SEQUENCE_ID = 1
VALIDATION_SEQUENCE_ID = 2

POSE_SUFFIX = 'pose'
RGB_SUFFIX = 'color'
DEPTH_SUFFIX = 'rendered.depth'

WIDTH = 540
HEIGHT = 960
CAMERA_TYPE = kapture.CameraType.PINHOLE

INTRINSICS = {
    'seq01_01': [756.026, 756.832, 270.419, 492.889],
    'seq01_02': [756.026, 756.832, 270.418, 492.889],
    'seq01_03': [756.026, 756.832, 270.419, 492.889],
    'seq01_04': [758.792, 760.052, 266.208, 485.694],
    'seq01_05': [759.611, 760.513, 268.301, 491.445],
    'seq01_06': [759.611, 760.513, 268.301, 491.445],
    'seq02_01': [756.026, 756.832, 270.419, 492.889],
    'seq02_02': [758.792, 760.051, 266.208, 485.693],
    'seq02_03': [756.026, 756.832, 270.419, 492.889],
    'seq02_04': [756.026, 756.832, 270.419, 492.889],
    'seq02_05': [755.581, 756.525, 269.186, 488.945],
    'seq02_06': [759.507, 760.646, 268.339, 487.215],
    'seq02_07': [759.507, 760.646, 268.339, 487.215],
    'seq02_08': [756.026, 756.832, 270.419, 492.889],
    'seq03_01': [756.026, 756.832, 270.419, 492.889],
    'seq03_02': [756.026, 756.832, 270.418, 492.889],
    'seq03_03': [756.026, 756.832, 270.419, 492.889],
    'seq03_04': [756.026, 756.832, 270.419, 492.889],
    'seq03_05': [756.026, 756.832, 270.419, 492.889],
    'seq03_06': [758.792, 760.052, 266.208, 485.694],
    'seq03_07': [756.026, 756.832, 270.419, 492.889],
    'seq04_01': [758.792, 760.052, 266.208, 485.694],
    'seq04_02': [756.026, 756.832, 270.418, 492.889],
    'seq04_03': [756.026, 756.832, 270.419, 492.889],
    'seq04_04': [756.026, 756.832, 270.419, 492.889],
    'seq04_05': [756.026, 756.832, 270.419, 492.889],
    'seq04_06': [756.026, 756.832, 270.419, 492.889],
    'seq04_07': [756.026, 756.832, 270.419, 492.889],
    'seq04_08': [756.026, 756.832, 270.419, 492.889],
    'seq04_09': [756.026, 756.832, 270.419, 492.889],
    'seq04_10': [760.420, 761.683, 268.748, 486.778],
    'seq05_01': [756.026, 756.832, 270.419, 492.889],
    'seq05_02': [760.420, 761.683, 268.747, 486.778],
    'seq05_03': [756.026, 756.832, 270.419, 492.889],
    'seq05_04': [756.026, 756.832, 270.419, 492.889],
    'seq05_05': [758.792, 760.052, 266.208, 485.694],
    'seq06_01': [756.026, 756.832, 270.419, 492.889],
    'seq06_02': [756.026, 756.832, 270.418, 492.889],
    'seq06_03': [756.026, 756.832, 270.419, 492.889],
    'seq06_04': [756.026, 756.832, 270.419, 492.889],
    'seq06_05': [756.026, 756.832, 270.419, 492.889],
    'seq06_06': [756.026, 756.832, 270.419, 492.889],
    'seq06_07': [756.026, 756.832, 270.419, 492.889],
    'seq06_08': [756.026, 756.832, 270.419, 492.889],
    'seq06_09': [756.026, 756.832, 270.419, 492.889],
    'seq06_10': [756.026, 756.832, 270.419, 492.889],
    'seq06_11': [756.026, 756.832, 270.419, 492.889],
    'seq06_12': [758.792, 760.052, 266.208, 485.694],
    'seq07_01': [760.420, 761.683, 268.748, 486.778],
    'seq07_02': [760.420, 761.683, 268.747, 486.778],
    'seq07_03': [760.420, 761.683, 268.748, 486.778],
    'seq07_04': [760.420, 761.683, 268.748, 486.778],
    'seq07_05': [759.611, 760.513, 268.301, 491.445],
    'seq07_06': [759.611, 760.513, 268.301, 491.445],
    'seq07_07': [759.507, 760.646, 268.339, 487.215],
    'seq07_08': [759.507, 760.646, 268.339, 487.215],
    'seq08_01': [759.507, 760.646, 268.339, 487.215],
    'seq08_02': [757.958, 759.273, 263.450, 490.819],
    'seq08_03': [759.507, 760.646, 268.339, 487.215],
    'seq08_04': [758.754, 759.788, 269.799, 489.091],
    'seq08_05': [759.507, 760.646, 268.339, 487.215],
    'seq09_01': [758.792, 760.052, 266.208, 485.694],
    'seq09_02': [758.792, 760.051, 266.208, 485.693],
    'seq09_03': [758.792, 760.052, 266.208, 485.694],
    'seq09_04': [758.792, 760.052, 266.208, 485.694],
    'seq09_05': [758.792, 760.052, 266.208, 485.694],
    'seq10_01': [756.026, 756.832, 270.419, 492.889],
    'seq10_02': [756.026, 756.832, 270.418, 492.889],
    'seq10_03': [756.026, 756.832, 270.419, 492.889],
    'seq10_04': [756.026, 756.832, 270.419, 492.889],
    'seq10_05': [756.026, 756.832, 270.419, 492.889],
    'seq10_06': [755.581, 756.525, 269.186, 488.945],
    'seq10_07': [756.026, 756.832, 270.419, 492.889],
    'seq10_08': [759.611, 760.513, 268.301, 491.445]
}


def _import_RIO10_sequence(sequence_root: str, sequence_list: List[str],
                           kapture_dir_path: str, images_import_method: TransferAction,
                           force_overwrite_existing: bool):
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    snapshots = kapture.RecordsCamera()
    depth_maps = kapture.RecordsDepth()
    trajectories = kapture.Trajectories()
    rigs = kapture.Rigs()
    sensors = kapture.Sensors()
    RIO10_seq_filename_re = re.compile(r'^frame-(?P<frame_id>\d{6})\.(?P<suffix>.*)\.(?P<ext>\w*)$')

    current_timestamp = 0
    for senquence_path in sequence_list:
        rgb_sensor_id = senquence_path + '_rgb'
        depth_sensor_id = senquence_path + '_depth'
        rig_sensor_id = senquence_path
        sensors[rgb_sensor_id] = kapture.Camera(CAMERA_TYPE, [WIDTH, HEIGHT] + INTRINSICS[senquence_path])
        sensors[depth_sensor_id] = kapture.Camera(CAMERA_TYPE, [WIDTH, HEIGHT] + INTRINSICS[senquence_path],
                                                  sensor_type=SENSOR_TYPE_DEPTH_CAM)
        rigs[rig_sensor_id, rgb_sensor_id] = kapture.PoseTransform()
        rigs[rig_sensor_id, depth_sensor_id] = kapture.PoseTransform()

        file_list = os.listdir(path.join(sequence_root, senquence_path))
        logger.debug(f'populating {senquence_path} files ...')
        RIO10_seq_filenames = {filename: RIO10_seq_filename_re.search(filename).groupdict()
                               for filename in sorted(file_list)
                               if RIO10_seq_filename_re.search(filename)}
        # reorg as shot[seq, id] = {color: , depth: , pose: , ...}
        shots = {}
        for timestamp, (filename, file_attribs) in enumerate(RIO10_seq_filenames.items()):
            shot_id = int(file_attribs['frame_id'])
            shots.setdefault(shot_id, {})[file_attribs['suffix']] = filename
        for timestamp, shot_id in enumerate(shots):
            shots[shot_id]['timestamp'] = shot_id + current_timestamp
        for shot in shots.values():
            snapshots[shot['timestamp'], rgb_sensor_id] = senquence_path + '/' + shot[RGB_SUFFIX]
            # kapture depth files are not png
            kapture_depth_map_filename = senquence_path + '/' + shot[DEPTH_SUFFIX][:-len('.png')]
            depth_maps[shot['timestamp'], depth_sensor_id] = kapture_depth_map_filename

            if POSE_SUFFIX in shot:
                pose_filepath = path.join(sequence_root, senquence_path, shot['pose'])
                pose_mat = np.loadtxt(pose_filepath)  # camera-to-world, 4×4 matrix in homogeneous coordinates
                with open(pose_filepath, 'r') as file:
                    if 'INF' in file.read():
                        timestamp = shot['timestamp']
                        image_name = shot[RGB_SUFFIX]
                        logger.debug(f'ts={timestamp}, name={image_name}: ignored inf pose')
                        continue
                rotation_mat = pose_mat[0:3, 0:3]
                position_vec = pose_mat[0:3, 3]
                rotation_quat = quaternion.from_rotation_matrix(rotation_mat)
                pose_world_from_cam = kapture.PoseTransform(r=rotation_quat, t=position_vec)
                pose_cam_from_world = pose_world_from_cam.inverse()
                trajectories[shot['timestamp'], rig_sensor_id] = pose_cam_from_world
        current_timestamp += max(shots.keys()) + 1 + TIMESTAMP_SEQUENCE_BREAK

    # import (copy) image files.
    logger.info('copying image files ...')
    image_filenames = snapshots.data_list()
    import_record_data_from_dir_auto(sequence_root, kapture_dir_path, image_filenames, images_import_method)

    # import (copy) depth map files.
    logger.info('converting depth files ...')
    depth_map_filenames = kapture.io.records.records_to_filepaths(depth_maps, kapture_dir_path)
    hide_progress = logger.getEffectiveLevel() > logging.INFO
    for depth_map_filename, depth_map_filepath_kapture in tqdm(depth_map_filenames.items(), disable=hide_progress):
        depth_map_filepath_rio10 = path.join(sequence_root, depth_map_filename + '.png')
        depth_map = np.array(Image.open(depth_map_filepath_rio10))
        # depth maps is in mm in rio10, convert it to meters
        depth_map = depth_map.astype(np.float32) * DEPTH_TO_METER
        kapture.io.records.depth_map_to_file(depth_map_filepath_kapture, depth_map)

    imported_kapture = kapture.Kapture(
        records_camera=snapshots,
        records_depth=depth_maps,
        rigs=rigs,
        trajectories=trajectories or None,
        sensors=sensors)
    logger.info('writing imported data ...')
    kapture_to_dir(kapture_dir_path, imported_kapture)


def import_RIO10(dRIO10_path: str,
                 kapture_dir_path: str,
                 force_overwrite_existing: bool = False,
                 images_import_method: TransferAction = TransferAction.skip
                 ) -> None:
    """
    Imports RIO10 dataset and save them as kapture.

    :param dRIO10_path: path to the RIO10 root path (contains folders named sceneXX)
    :param kapture_dir_path: path to kapture output top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files
    """
    os.makedirs(kapture_dir_path, exist_ok=True)

    scenes = list(range(1, 11))
    for scene_id in scenes:
        logger.info(f'import scene {scene_id}...')
        scene_path = path.join(dRIO10_path, f'scene{scene_id:02d}')
        if not path.isdir(scene_path):
            logger.warning(f'{scene_path} does not exist')
            continue
        seq_path = path.join(scene_path, f'seq{scene_id:02d}')
        if not path.isdir(seq_path):
            logger.warning(f'{seq_path} does not exist')
            continue
        seq_list = os.listdir(seq_path)

        seq_subfolder_re = re.compile(r'seq(?P<scene_id>\d{2})_(?P<seq_id>\d{2})')

        seq_subfolders = {}
        for seq_subfolder in seq_list:
            seq_subfolder_search = seq_subfolder_re.search(seq_subfolder)
            if seq_subfolder_search:
                seq_subfolder_re_dict = seq_subfolder_search.groupdict()
                if int(seq_subfolder_re_dict['scene_id']) != scene_id:
                    logger.warning(f'{seq_path} : found a sequence with wrong sequence id, skipped')
                    continue
                seq_subfolders[int(seq_subfolder_re_dict['seq_id'])] = seq_subfolder
        if MAPPING_SEQUENCE_ID in seq_subfolders:
            # 1 is always the mapping sequence
            _import_RIO10_sequence(seq_path, [seq_subfolders[MAPPING_SEQUENCE_ID]],
                                   path.join(kapture_dir_path, f'scene{scene_id:02d}', 'mapping'),
                                   images_import_method, force_overwrite_existing)
        if VALIDATION_SEQUENCE_ID in seq_subfolders:
            # 2 is always the mapping sequence
            _import_RIO10_sequence(seq_path, [seq_subfolders[VALIDATION_SEQUENCE_ID]],
                                   path.join(kapture_dir_path, f'scene{scene_id:02d}', 'validation'),
                                   images_import_method, force_overwrite_existing)
        testing_sequences = [v
                             for k, v in sorted(seq_subfolders.items())
                             if k != MAPPING_SEQUENCE_ID and k != VALIDATION_SEQUENCE_ID]
        if len(testing_sequences) > 0:
            _import_RIO10_sequence(seq_path, testing_sequences,
                                   path.join(kapture_dir_path, f'scene{scene_id:02d}', 'testing'),
                                   images_import_method, force_overwrite_existing)


def import_RIO10_command_line() -> None:
    """
    Imports RGB-D Dataset RIO10 and save them as kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='Imports RGB-D Dataset RIO10 files to the kapture format.')
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
    parser.add_argument('-i', '--input', required=True, help='input path to RIO10 root path')
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

    import_RIO10(dRIO10_path=args.input,
                 kapture_dir_path=args.output,
                 force_overwrite_existing=args.force,
                 images_import_method=args.image_transfer)


if __name__ == '__main__':
    import_RIO10_command_line()
