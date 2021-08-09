#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a 12scenes model into a kapture.
The RGB-D Dataset 12-Scenes data structure is defined here:
    http://graphics.stanford.edu/projects/reloc/

Each sequence contains:
    Color frames (frame-XXXXXX.color.jpg): RGB, 24-bit, JPG
    Depth frames (frame-XXXXXX.depth.png): depth (mm), 16-bit, PNG (invalid depth is set to 0)
    Camera poses (frame-XXXXXX.pose.txt): camera-to-world
    Camera calibration (info.txt): color and depth camera intrinsics and extrinsics.
    Note that these are the default intrinsics and we did not perform any calibration.
"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from typing import Optional
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.utils.paths import path_secure

logger = logging.getLogger('12scenes')
MODEL = kapture.CameraType.RADIAL

POSE_SUFFIX = 'pose'
RGB_SUFFIX = 'color'
DEPTH_SUFFIX = 'depth'
RGB_SENSOR_ID = 'ipad_camera'
DEPTH_SENSOR_ID = 'structure_io_depth_camera'
REG_DEPTH_SENSOR_ID = 'structure_io_depth_camera_reg'
RGBD_SENSOR_ID = 'rgbd_rig'


def get_camera_matrix(fx: float, fy: float, cx: float, cy: float):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def get_K(camera_type: kapture.CameraType, camera_params):
    if camera_type == kapture.CameraType.UNKNOWN_CAMERA:
        width = camera_params[0]
        height = camera_params[1]
        focal = 1.2 * max(width, height)
        return get_camera_matrix(focal, focal, camera_params[0] / 2, camera_params[1] / 2)
    elif camera_type in [kapture.CameraType.PINHOLE, kapture.CameraType.OPENCV,
                         kapture.CameraType.OPENCV_FISHEYE, kapture.CameraType.FULL_OPENCV,
                         kapture.CameraType.FOV, kapture.CameraType.THIN_PRISM_FISHEYE]:
        return get_camera_matrix(camera_params[2], camera_params[3], camera_params[4], camera_params[5])
    else:
        return get_camera_matrix(camera_params[2], camera_params[3], camera_params[4], camera_params[4])


def dilate(x, k):
    m, n = x.shape
    y = np.empty_like(x)
    for i in range(0, m):
        for j in range(0, n):
            currmax = x[i, j]
            for ii in range(int(max(0, i-k/2)), int(min(m, i+k/2+1))):
                for jj in range(int(max(0, j-k/2)), int(min(n, j+k/2+1))):
                    elt = x[ii, jj]
                    if elt > currmax:
                        currmax = elt
            y[i, j] = currmax
    return y


def register_depth(Kdepth, Kcolor, Rt, depth, width_color, height_color):
    reg_depth = np.zeros((height_color, width_color), dtype=np.float32)
    y, x = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    z = depth.reshape(1, -1)
    x = (x - Kdepth[0, 2]) / Kdepth[0, 0]
    y = (y - Kdepth[1, 2]) / Kdepth[1, 1]
    pts = np.vstack((x * z, y * z, z))
    pts = Rt[:3, :3] @ pts + Rt[:3, 3:]
    pts = Kcolor @ pts
    px = np.round(pts[0, :] / pts[2, :])
    py = np.round(pts[1, :] / pts[2, :])
    mask = (px >= 0) * (py >= 0) * (px < width_color) * (py < height_color)
    reg_depth[py[mask].astype(int), px[mask].astype(int)] = pts[2, mask]

    reg_depth = ndimage.grey_dilation(reg_depth, size=(3, 3))

    return reg_depth


def import_12scenes(d12scenes_path: str,
                    kapture_dir_path: str,
                    force_overwrite_existing: bool = False,
                    images_import_method: TransferAction = TransferAction.skip,
                    partition: Optional[str] = None
                    ) -> None:
    """
    Imports RGB-D Dataset 12-Scenes dataset and save them as kapture.

    :param d12scenes_path: path to the 12scenes sequence root path
    :param kapture_dir_path: path to kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    :param partition: if specified = 'mapping' or 'query'.
    """
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    logger.info('loading all content ...')

    d7s_filename_re = re.compile(r'frame-(?P<frame_id>\d{6})\.(?P<suffix>\w*)\.(?P<ext>\w*)')

    # populate all relevant files
    d12images_path = os.path.join(d12scenes_path, 'data')
    d7s_filenames = (path_secure(path.relpath(path.join(dp, fn), d12images_path))
                     for dp, _, fs in os.walk(d12images_path) for fn in fs)

    logger.info('populating 12-scenes files ...')
    d7s_filenames = {filename: d7s_filename_re.search(filename).groupdict()
                     for filename in sorted(d7s_filenames)
                     if d7s_filename_re.search(filename)}

    # reorg as shot[seq, id] = {color: , depth: , pose: , ...}
    shots = {}
    for timestamp, (filename, file_attribs) in enumerate(d7s_filenames.items()):
        shot_id = int(file_attribs['frame_id'])
        shots.setdefault(shot_id, {})[file_attribs['suffix']] = filename

    # fake timestamps
    for timestamp, shot_id in enumerate(shots):
        shots[shot_id]['timestamp'] = timestamp

    # if given, filter partition
    if partition is not None:
        # read the authors split file
        partition_filepath = path.join(d12scenes_path, 'split.txt')
        if not path.isfile(partition_filepath):
            raise FileNotFoundError(f'partition file is missing: {partition_filepath}.')

        with open(partition_filepath, 'rt') as file:
            # note from dsac++; the first sequence is used for testing, everything else for training
            d7s_split_exp = r'^sequence(?P<sequence>\d+) \[frames=(?P<count>\d+)\]  \[start=(?P<start_frame>\d+) ;' \
                            r' end=(?P<end_frame>\d+)\]$'
            d7s_split_re = re.compile(d7s_split_exp)
            split_sequences = [re.match(d7s_split_re, line) for line in file.readlines()]
            if len(split_sequences) < 1 or not split_sequences[0]:
                raise ValueError('failed to parse split.txt file')
            test_split = (int(split_sequences[0].group('start_frame')), int(split_sequences[0].group('end_frame')))

            # filter out
            if partition == "query":
                shots = {frame: shot
                         for frame, shot in shots.items()
                         if test_split[0] <= frame <= test_split[1]
                         }
            elif partition == "mapping":
                shots = {frame: shot
                         for frame, shot in shots.items()
                         if frame < test_split[0] or frame > test_split[1]
                         }
            else:
                raise ValueError('invalid partition name')

    if len(shots) == 0:
        raise FileNotFoundError('no file found: make sure the path to 12scenes sequence is valid.')

    # eg. shots['000000'] =
    #       {
    #           'color': 'seq-01/frame-000000.color.jpg',
    #           'depth': 'seq-01/frame-000000.depth.png',
    #           'pose': 'seq-01/frame-000000.pose.txt',
    #           'timestamp': 0}

    # images + depth maps
    logger.info('populating image and depth maps files ...')
    snapshots = kapture.RecordsCamera()
    depth_maps = kapture.RecordsDepth()
    for shot in shots.values():
        snapshots[shot['timestamp'], RGB_SENSOR_ID] = shot['color']
        kapture_depth_map_filename = shot['depth'][:-len('.png')]  # kapture depth files are not png
        depth_maps[shot['timestamp'], DEPTH_SENSOR_ID] = kapture_depth_map_filename
        kapture_registered_depth_map_filename = shot['depth'][:-len('.png')] + '.reg'  # kapture depth files are not png
        depth_maps[shot['timestamp'], REG_DEPTH_SENSOR_ID] = kapture_registered_depth_map_filename

    # poses
    logger.info('import poses files ...')
    trajectories = kapture.Trajectories()
    for shot in shots.values():
        pose_filepath = path.join(d12images_path, shot['pose'])
        pose_mat = np.loadtxt(pose_filepath)  # camera-to-world, 4×4 matrix in homogeneous coordinates
        with open(pose_filepath, 'r') as file:
            if 'INF' in file.read():
                timestamp = shot['timestamp']
                image_name = shot['color']
                logger.debug(f'ts={timestamp}, name={image_name}: ignored inf pose')
                continue
        rotation_mat = pose_mat[0:3, 0:3]
        position_vec = pose_mat[0:3, 3]
        rotation_quat = quaternion.from_rotation_matrix(rotation_mat)
        pose_world_from_cam = kapture.PoseTransform(r=rotation_quat, t=position_vec)
        pose_cam_from_world = pose_world_from_cam.inverse()
        trajectories[shot['timestamp'], RGBD_SENSOR_ID] = pose_cam_from_world

    # sensors
    """
    Read info.txt
    """
    info_filepath = path.join(d12scenes_path, 'info.txt')
    if not path.isfile(info_filepath):
        raise FileNotFoundError(f'info file is missing: {info_filepath}.')

    with open(info_filepath, 'rt') as file:
        info_dict = {}
        for line in file.readlines():
            line_splits = line.rstrip().split(' = ')
            info_dict[line_splits[0]] = line_splits[1]

    sensors = kapture.Sensors()
    camera_type = kapture.CameraType.PINHOLE
    assert 'm_calibrationColorIntrinsic' in info_dict
    assert 'm_colorWidth' in info_dict
    assert 'm_colorHeight' in info_dict
    rgb_intrinsics = [float(v) for v in info_dict['m_calibrationColorIntrinsic'].split(' ')]
    # w, h, fx, fy, cx, cy
    rgb_camera_params = [int(info_dict['m_colorWidth']), int(info_dict['m_colorHeight']),
                         rgb_intrinsics[0], rgb_intrinsics[5], rgb_intrinsics[2], rgb_intrinsics[6]]
    sensors[RGB_SENSOR_ID] = kapture.Camera(
        name=RGB_SENSOR_ID,
        camera_type=camera_type,
        camera_params=rgb_camera_params
    )

    assert 'm_calibrationDepthIntrinsic' in info_dict
    assert 'm_depthWidth' in info_dict
    assert 'm_depthHeight' in info_dict
    depth_intrinsics = [float(v) for v in info_dict['m_calibrationDepthIntrinsic'].split(' ')]
    # w, h, fx, fy, cx, cy
    depth_camera_params = [int(info_dict['m_depthWidth']), int(info_dict['m_depthHeight']),
                           depth_intrinsics[0], depth_intrinsics[5], depth_intrinsics[2], depth_intrinsics[6]]
    sensors[DEPTH_SENSOR_ID] = kapture.Camera(
        name=DEPTH_SENSOR_ID,
        camera_type=camera_type,
        camera_params=depth_camera_params,
        sensor_type='depth'
    )

    sensors[REG_DEPTH_SENSOR_ID] = kapture.Camera(
        name=REG_DEPTH_SENSOR_ID,
        camera_type=camera_type,
        camera_params=rgb_camera_params,
        sensor_type='depth'
    )

    # bind camera and depth sensor into a rig
    logger.info('building rig with camera and depth sensor ...')
    rigs = kapture.Rigs()
    rigs[RGBD_SENSOR_ID, RGB_SENSOR_ID] = kapture.PoseTransform()
    rigs[RGBD_SENSOR_ID, DEPTH_SENSOR_ID] = kapture.PoseTransform()
    rigs[RGBD_SENSOR_ID, REG_DEPTH_SENSOR_ID] = kapture.PoseTransform()

    # import (copy) image files.
    logger.info('copying image files ...')
    image_filenames = snapshots.data_list()
    import_record_data_from_dir_auto(d12images_path, kapture_dir_path, image_filenames, images_import_method)

    # import (copy) depth map files.
    logger.info('converting depth files ...')
    depth_map_filenames = kapture.io.records.records_to_filepaths(depth_maps, kapture_dir_path)
    hide_progress = logger.getEffectiveLevel() > logging.INFO
    for depth_map_filename, depth_map_filepath_kapture in tqdm(depth_map_filenames.items(), disable=hide_progress):
        if '.reg' in depth_map_filename:
            continue
        depth_map_filepath_12scenes = path.join(d12images_path, depth_map_filename + '.png')
        depth_map = np.array(Image.open(depth_map_filepath_12scenes))
        # depth maps is in mm in 12scenes, convert it to meters
        depth_map = depth_map.astype(np.float32) * 1.0e-3
        kapture.io.records.depth_map_to_file(depth_map_filepath_kapture, depth_map)
        # register depth to rgb
        reg_depth_map = register_depth(get_K(camera_type, depth_camera_params), get_K(camera_type, rgb_camera_params),
                                       np.eye(4), depth_map, rgb_camera_params[0], rgb_camera_params[1])
        kapture.io.records.depth_map_to_file(depth_map_filepath_kapture + '.reg', reg_depth_map)

    # pack into kapture format
    imported_kapture = kapture.Kapture(
        records_camera=snapshots,
        records_depth=depth_maps,
        rigs=rigs,
        trajectories=trajectories,
        sensors=sensors)

    logger.info('writing imported data ...')
    kapture_to_dir(kapture_dir_path, imported_kapture)


def import_12scenes_command_line() -> None:
    """
    Imports RGB-D Dataset 12-Scenes dataset and save them as kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='Imports RGB-D Dataset 12-Scenes files to the kapture format.')
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
                        help='input path Dataset 12-Scenes sequence root path')
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

    import_12scenes(d12scenes_path=args.input,
                    kapture_dir_path=args.output,
                    force_overwrite_existing=args.force,
                    images_import_method=args.image_transfer,
                    partition=args.partition)


if __name__ == '__main__':
    import_12scenes_command_line()
