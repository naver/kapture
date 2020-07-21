#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import the Scape-Imperial Localisation Dataset dataset to the kapture format.

See research page at https://research.scape.io/silda/
and dataset description there: https://github.com/abmmusa/silda.
"""

import argparse
import logging
import re
import os
import os.path as path
import numpy as np
import quaternion
from typing import Dict, List, Union
from tqdm import tqdm
import path_to_kapture
import kapture
import kapture.utils.logging
import kapture.io.csv as csv
import kapture.io.structure
from kapture.io.records import TransferAction, get_image_fullpath, transfer_files_from_dir
from kapture.utils.paths import path_secure
import kapture.io.features

logger = logging.getLogger('silda')

SILDA_IMAGE_NAME_PATTERN = re.compile(
    r'(?P<corpus>(train)|(query))/(?P<basename>(?P<timestamp>\d+)_(?P<cam_id>\d+)\.png)')
SILDA_IMAGE_SIZE = np.array([1024, 1024])
SILDA_CALIB_IMAGE_SIZE = np.array([2496, 2496])


def import_silda(silda_dirpath: str,
                 destination_kapture_dirpath: str,
                 fallback_cam_model: str = 'FOV',
                 do_split_cams: bool = False,
                 corpus: List[str] = ['train', 'query'],
                 replace_pose_rig: bool = False,
                 force_overwrite_existing: bool = False,
                 images_import_strategy: TransferAction = TransferAction.link_absolute) -> None:
    """
    Imports data from silda dataset.

    :param silda_dirpath: path to the silda top directory
    :param destination_kapture_dirpath: input path to kapture directory.
    :param fallback_cam_model: camera model to fallback when necessary
    :param do_split_cams: If true, re-organises and renames the image files to split apart cameras.
    :param corpus: the list of corpus to be imported, among 'train', 'query'.
    :param replace_pose_rig: if True, replaces poses of individual cameras with poses of the rig.
    :param force_overwrite_existing: if true, Silently overwrite kapture files if already exists.
    :param images_import_strategy: how to copy image files.
    """

    # sanity check
    if TransferAction.root_link == images_import_strategy and do_split_cams:
        raise ValueError('impossible to only link images directory and applying split cam.')
    hide_progress_bars = logger.getEffectiveLevel() >= logging.INFO

    # prepare output directory
    kapture.io.structure.delete_existing_kapture_files(destination_kapture_dirpath, force_overwrite_existing)
    os.makedirs(destination_kapture_dirpath, exist_ok=True)

    # images ###########################################################################################################
    logger.info('Processing images ...')
    # silda-images
    # ├── query
    # └── train
    #       ...
    #       ├── 1445_0.png
    #       ├── 1445_1.png
    #       ...
    silda_images_root_path = path.join(silda_dirpath, 'silda-images')
    # list all png files (its PNG in silda) using a generator.
    silda_images_crawler = (path_secure(path.join(dirpath, filename))
                            for dirpath, sd, fs in os.walk(silda_images_root_path)
                            for filename in fs
                            if filename.endswith('.png'))

    # restrain to requested corpus only (query and/or train)
    if len(corpus) == 2:  # ['train', 'query'] == None
        corpus = None

    # if corpus specified, filter by those which directory name match corpus.
    if corpus:
        logger.debug(f'only importing {corpus} part.')
        silda_images_crawler = (filepath for filepath in silda_images_crawler
                                if path.basename(path.dirname(filepath)) in corpus)

    silda_image_filenames, kapture_image_filenames = [], []
    snapshots = kapture.RecordsCamera()
    silda_name_to_ids = {}
    for original_image_filepath in tqdm(silda_images_crawler, disable=hide_progress_bars):
        # retrieve info from image filename
        file_info: Dict[str, Union[int, str]]  # To avoid warnings about type of the value
        file_info = SILDA_IMAGE_NAME_PATTERN.search(original_image_filepath).groupdict()
        file_info['timestamp'] = int(file_info['timestamp'])
        silda_image_filename = path.join(file_info['corpus'], file_info['basename'])
        # original_image_filepath = 'silda-images/query/1445_0.png'
        # info = {'corpus': 'query ', 'basename': '1445_0.png', 'timestamp': 1445, 'cam_id': '0'}
        # create a path of the image into NLE dir
        if do_split_cams:
            # re-organise images with subfolders per corpus/camera/timestamp.png
            kapture_image_filename = path.join(file_info['corpus'],
                                               file_info['cam_id'],
                                               '{:04d}.png'.format(file_info['timestamp']))
        else:
            # keep the original file hierarchy
            kapture_image_filename = silda_image_filename
        silda_image_filenames.append(silda_image_filename)
        kapture_image_filenames.append(kapture_image_filename)
        snapshots[file_info['timestamp'], file_info['cam_id']] = kapture_image_filename
        silda_name_to_ids[file_info['basename']] = (file_info['timestamp'], file_info['cam_id'])

    # intrinsics #######################################################################################################
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
        intrinsic_filepath = path.join(silda_dirpath, 'camera-intrinsics', random_image_intrinsic)
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

    # extrinsics #######################################################################################################
    logger.info('Processing trajectories ...')
    trajectories = kapture.Trajectories()
    with open(path.join(silda_dirpath, 'silda-train-poses.txt')) as file:
        lines = file.readlines()
        lines = (line.rstrip().split() for line in lines)
        extrinsics = {
            line[0]: np.array(line[1:8], dtype=np.float) for line in lines
        }

    for silda_image_name, pose_params in tqdm(extrinsics.items(), disable=hide_progress_bars):
        # Silda poses are 7-dim vectors with the rotation quaternion,
        # and the translation vector. The order needs to be:
        # qw,qx,qy,qz,tx,ty,tz
        # The parameters should be described in terms of camera to world transformations
        if silda_image_name not in silda_name_to_ids:
            # if this is not referenced: means its part of the corpus to be ignored.
            continue
        pose = kapture.PoseTransform(pose_params[0:4], pose_params[4:7]).inverse()
        timestamp, cam_id = silda_name_to_ids[silda_image_name]
        trajectories[timestamp, cam_id] = pose

    # rigs
    logger.info('Making up a rig ...')
    rigs = kapture.Rigs()
    pose_babord = kapture.PoseTransform(t=[0, 0, 0], r=quaternion.from_rotation_vector([0, -np.pi / 2, 0]))
    pose_tribord = kapture.PoseTransform(t=[0, 0, 0], r=quaternion.from_rotation_vector([0, np.pi / 2, 0]))
    rigs['silda_rig', '0'] = pose_babord
    rigs['silda_rig', '1'] = pose_tribord
    if replace_pose_rig:
        logger.info('replacing camera poses with rig poses.')
        kapture.rigs_recover_inplace(trajectories, rigs)

    # pack it all together
    kapture_data = kapture.Kapture(
        sensors=cameras,
        records_camera=snapshots,
        trajectories=trajectories,
        rigs=rigs
    )

    logger.info('saving to Kapture  ...')
    kapture.io.csv.kapture_to_dir(destination_kapture_dirpath, kapture_data)

    # finally import images
    logger.info('importing image files  ...')
    if images_import_strategy != TransferAction.skip:
        # importing image files
        logger.info(f'importing {len(silda_image_filenames)} images ...')
        assert len(silda_image_filenames) == len(kapture_image_filenames)
        silda_image_filepaths = (
            path.join(silda_images_root_path, record_filename)
            for record_filename in silda_image_filenames)
        kapture_image_filepaths = (
            get_image_fullpath(destination_kapture_dirpath, record_filename)
            for record_filename in kapture_image_filenames)
        transfer_files_from_dir(silda_image_filepaths, kapture_image_filepaths, images_import_strategy)
    logger.info('done.')


def import_silda_command_line():
    """
    Do the SILDa to kapture import using the command line parameters provided by the user.
    """
    parser = argparse.ArgumentParser(description='imports SILDa dataset to kapture format.')
    ####################################################################################################################
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete kapture if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i', '--input',
                        help='path to silda root directory.')
    parser.add_argument('-o', '--output', required=True,
                        help='output directory where to save NLE files.')
    parser.add_argument('-s', '--split_cams', action='store_true', default=False,
                        help='reorganises the image file per camera folders.')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('--corpus', choices=['train', 'query'], nargs='+', default=['train', 'query'],
                        help='restrain (or not) do only train or query.')
    parser.add_argument('--cam_model', choices=['OPENCV_FISHEYE', 'FOV'], default='FOV',
                        help='camera model to be used.')
    parser.add_argument('--rig_collapse', action='store_true', default=False,
                        help='Replace camera poses with rig poses.')
    ####################################################################################################################
    args = parser.parse_args()
    logger.setLevel(args.verbose)

    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_silda(args.input,
                 args.output,
                 fallback_cam_model=args.cam_model,
                 do_split_cams=args.split_cams,
                 corpus=args.corpus,
                 replace_pose_rig=args.rig_collapse,
                 force_overwrite_existing=args.force,
                 images_import_strategy=args.image_transfer)


if __name__ == '__main__':
    import_silda_command_line()
