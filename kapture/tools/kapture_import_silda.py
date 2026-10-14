#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import the Scape-Imperial Localisation Dataset dataset to the kapture format.

See research page at https://research.scape.io/silda/
and dataset description there: https://github.com/abmmusa/silda.
"""

import logging
import re
import os
import os.path as path
import numpy as np
import quaternion
from typing import Any, Dict, Literal, Optional
from tqdm import tqdm
import kapture
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY
import kapture.io.csv
import kapture.io.structure
from kapture.io.binary import TransferAction, transfer_files_from_dir
from kapture.io.records import get_image_fullpath
from kapture.utils.paths import path_secure
import kapture.io.features

logger = logging.getLogger('silda')
SILDA_IMAGE_NAME_PATTERN = re.compile(r'(?P<filename>(?P<timestamp>\d+)_(?P<cam_id>\d+)\.png)')

SILDA_IMAGE_SIZE = np.array([1024, 1024])
SILDA_CALIB_IMAGE_SIZE = np.array([2496, 2496])
SILDA_CORPUS_SPLIT_FILENAMES = {
    'mapping': 'train_imgs.txt',
    'query': 'query_imgs.txt',
}


def _import_cameras(silda_dir_path, snapshots, fallback_cam_model) -> kapture.Sensors:
    logger.info('Processing sensors ...')
    cameras = kapture.Sensors()
    # use hard coded intrinsics
    # evaluated using colmap
    # 1 OPENCV_FISHEYE 1024 1024 393.299 394.815 512 512 -0.223483 0.117325 -0.0326138 0.00361082
    #                  fx, fy, cx, cy, omega
    # 1 FOV 1024 1024 300 300 512 512 0.899632
    cam_id_list = sorted(set(cam_id for _, cam_id, _ in kapture.flatten(snapshots)))
    for cam_id in cam_id_list:
        # pick a image for that cam id
        random_image_intrinsic = next(f'{timestamp}_{cam_id}.intrinsics'  # keep only filename (thats what silda expect)
                                      for timestamp, cid, filename in kapture.flatten(snapshots)
                                      if cid == cam_id)
        logger.debug(f'camera {cam_id} intrinsics : picking at random: ("{random_image_intrinsic}")')
        intrinsic_filepath = path.join(silda_dir_path, 'camera-intrinsics', random_image_intrinsic)
        logger.debug(f'loading file: "{intrinsic_filepath}"')
        silda_proj_params = np.loadtxt(intrinsic_filepath)
        # only retrieve principal point from intrinsics,
        # because the rest correspond to a fisheye model not available in colmap.
        principal_point = (silda_proj_params[0:2] * SILDA_IMAGE_SIZE).flatten().tolist()
        projection = fallback_cam_model
        if 'OPENCV_FISHEYE' == projection:
            focal_length = [393.299, 394.815]
            fisheye_coefficients = [-0.223483,
                                    0.117325, -0.0326138, 0.00361082]
            #          //    fx, fy, cx, cy, k1, k2, k3, k4
            proj_params = focal_length + principal_point + fisheye_coefficients
        elif 'FOV' == projection:
            # use hard coded intrinsics from Torsten reconstruction, ie. :
            #       217.294036, 217.214703, 512.000000, 507.897400, -0.769113
            focal_length = [217.294036, 217.214703]
            # principal_point = [512.000000, 507.897400]
            omega = [-0.769113]
            #                  fx, fy, cx, cy, omega
            proj_params = focal_length + principal_point + omega
        else:
            raise ValueError('Only accepts OPENCV_FISHEYE, or FOV as projection model.')

        camera = kapture.Camera(projection, SILDA_IMAGE_SIZE.tolist() + proj_params)
        cameras[cam_id] = camera
    return cameras


def _import_trajectories(silda_dir_path, image_name_to_ids, hide_progress_bars) -> kapture.Trajectories:
    logger.info('Processing trajectories ...')
    trajectories = kapture.Trajectories()
    with open(path.join(silda_dir_path, 'silda-train-poses.txt')) as file:
        lines = file.readlines()
        lines = (line.rstrip().split() for line in lines)
        extrinsics = {
            line[0]: np.array(line[1:8], dtype=float) for line in lines
        }
    for silda_image_name, pose_params in tqdm(extrinsics.items(), disable=hide_progress_bars):
        # Silda poses are 7-dim vectors with the rotation quaternion,
        # and the translation vector. The order needs to be:
        # qw,qx,qy,qz,tx,ty,tz
        # The parameters should be described in terms of camera to world transformations
        if silda_image_name not in image_name_to_ids:
            # if this is not referenced: means its part of the corpus to be ignored.
            continue
        pose = kapture.PoseTransform(pose_params[0:4], pose_params[4:7]).inverse()
        timestamp, cam_id = image_name_to_ids[silda_image_name]
        trajectories[timestamp, cam_id] = pose
    # if query, trajectories is empty, so juste do not save it
    if len(trajectories) == 0:
        trajectories = None
    return trajectories


def _make_rigs(replace_pose_rig, trajectories) -> kapture.Rigs:
    logger.info('Making up a rig ...')
    rigs = kapture.Rigs()
    pose_babord = kapture.PoseTransform(t=[0, 0, 0], r=quaternion.from_rotation_vector([0, -np.pi / 2, 0]))
    pose_tribord = kapture.PoseTransform(t=[0, 0, 0], r=quaternion.from_rotation_vector([0, np.pi / 2, 0]))
    rigs['silda_rig', '0'] = pose_babord
    rigs['silda_rig', '1'] = pose_tribord
    if replace_pose_rig:
        logger.info('replacing camera poses with rig poses.')
        kapture.rigs_recover_inplace(trajectories, rigs)
    return rigs


def import_silda(input: str,
                 output: str,
                 cam_model: Literal['OPENCV_FISHEYE', 'FOV'] = 'FOV',
                 split_cams: bool = False,
                 corpus: Optional[Literal['mapping', 'query']] = None,
                 rig_collapse: bool = False,
                 force: bool = False,
                 image_transfer: TransferAction = TransferAction.link_absolute,
                 verbose: Verbosity = DEFAULT_VERBOSITY,
                 quiet: bool = False) -> None:
    """
    Imports SILDa dataset to kapture format.

    :param input: path to the silda top directory
    :param output: path to the output kapture directory.
    :param cam_model: camera model to fallback when necessary
    :param split_cams: re-organises and renames the image files to split apart cameras.
    :param corpus: restrain to mapping or query images only.
    :param rig_collapse: replace camera poses with rig poses.
    :param force: silently overwrite kapture files if already exists.
    :param image_transfer: how to copy image files.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)

    # sanity check
    input = path_secure(path.abspath(input))
    output = path_secure(path.abspath(output))
    if TransferAction.root_link == image_transfer and split_cams:
        raise ValueError('impossible to only link images directory and applying split cam.')
    hide_progress_bars = logger.getEffectiveLevel() >= logging.INFO

    # prepare output directory
    kapture.io.structure.delete_existing_kapture_files(output, force)
    os.makedirs(output, exist_ok=True)

    # images ###########################################################################################################
    logger.info('Processing images ...')
    # silda-images
    #   ...
    #   ├── 1445_0.png
    #   ├── 1445_1.png
    #   ...
    silda_images_root_path = path.join(input, 'silda-images')
    # list all png files (its PNG in silda) using a generator.
    if corpus is not None:
        assert corpus in SILDA_CORPUS_SPLIT_FILENAMES
        # if corpus specified, filter by those which directory name match corpus.
        logger.debug(f'only importing {corpus} part.')
        corpus_file_path = path.join(input, SILDA_CORPUS_SPLIT_FILENAMES[corpus])
        with open(corpus_file_path, 'rt') as corpus_file:
            corpus_filenames = corpus_file.readlines()
            image_filenames_original = sorted(
                filename.strip()
                for filename in corpus_filenames
            )
    else:
        image_filenames_original = sorted(
            filename
            for dir_path, sd, fs in os.walk(silda_images_root_path)
            for filename in fs
            if filename.endswith('.png'))

    image_filenames_kapture = []
    snapshots = kapture.RecordsCamera()
    image_name_to_ids = {}  # '1445_0.png' -> (1445, 0)
    for image_filename_original in tqdm(image_filenames_original, disable=hide_progress_bars):
        # retrieve info from image filename
        name_parts_match = SILDA_IMAGE_NAME_PATTERN.match(image_filename_original)
        assert name_parts_match is not None
        shot_info: Dict[str, Any]
        shot_info = name_parts_match.groupdict()
        shot_info['timestamp'] = int(shot_info['timestamp'])  # To avoid warnings about type of the value
        # eg. file_info = {'filename': '1445_0.png', 'timestamp': 1445, 'cam_id': '0'}
        # create a path of the image into NLE dir
        if split_cams:
            # re-organise images with subfolders per corpus/camera/timestamp.png
            kapture_image_filename = path.join(shot_info['cam_id'],
                                               '{:04d}.png'.format(shot_info['timestamp']))
        else:
            # keep the original file hierarchy
            kapture_image_filename = image_filename_original

        image_filenames_kapture.append(kapture_image_filename)
        snapshots[shot_info['timestamp'], shot_info['cam_id']] = kapture_image_filename
        image_name_to_ids[shot_info['filename']] = (shot_info['timestamp'], shot_info['cam_id'])

    assert len(image_filenames_kapture) == len(image_filenames_original)
    # intrinsics #######################################################################################################
    cameras = _import_cameras(input, snapshots, cam_model)

    # extrinsics #######################################################################################################
    trajectories = _import_trajectories(input, image_name_to_ids, hide_progress_bars)

    # rigs
    rigs = _make_rigs(rig_collapse, trajectories)

    # pack it all together
    kapture_data = kapture.Kapture(
        sensors=cameras,
        records_camera=snapshots,
        trajectories=trajectories,
        rigs=rigs
    )

    logger.info('saving to Kapture  ...')
    kapture.io.csv.kapture_to_dir(output, kapture_data)

    # finally import images
    if image_transfer != TransferAction.skip:
        # importing image files
        logger.info(f'importing {len(image_filenames_original)} images ...')
        assert len(image_filenames_original) == len(image_filenames_kapture)
        image_file_paths_original = [
            path.join(silda_images_root_path, image_filename_kapture)
            for image_filename_kapture in image_filenames_original]
        image_file_paths_kapture = [
            get_image_fullpath(output, image_filename_kapture)
            for image_filename_kapture in image_filenames_kapture]
        transfer_files_from_dir(image_file_paths_original, image_file_paths_kapture, image_transfer)
    logger.info('done.')


def main():
    run_cli(import_silda, aliases={'--split_cams': ['-s']})


if __name__ == '__main__':
    main()
