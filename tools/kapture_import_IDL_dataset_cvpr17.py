#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports an IDL dataset cvpr17 to the kapture format.


    A Dataset for Benchmarking Image-based Localization
    http://openaccess.thecvf.com/content_cvpr_2017/html/Sun_A_Dataset_for_CVPR_2017_paper.html

    Xun Sun, Yuanfan Xie, Pei Luo, Liang Wang
    Baidu Autonomous Driving Business Unit

    https://sites.google.com/site/xunsunhomepage/

    Data format:
    =================

    The camera poses are saved in the following format:

    fx 0 cx
    0 fy cy
    0 0 1
    0 0 0
    R'
    cop
    w h

"""

import os
import os.path as path
import logging
import argparse
from PIL import Image
import quaternion
from typing import Union

# kapture
import path_to_kapture
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.utils.paths import path_secure


logger = logging.getLogger('IDL_dataset_cvpr17')


def import_idl_dataset_cvpr17(idl_dataset_path: str,
                              gt_path: Union[str, None],
                              kapture_path: str,
                              force_overwrite_existing: bool = False,
                              images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Reads the IDL dataset and copy it to a kapture.

    :param idl_dataset_path: path to the IDL dataset
    :param gt_path: ground truth data path
    :param kapture_path: path to the kapture top directory to create
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """

    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()

    file_list = [os.path.relpath(os.path.join(dirpath, filename), idl_dataset_path)
                 for dirpath, dirs, filenames in os.walk(idl_dataset_path)
                 for filename in filenames]
    file_list = sorted(file_list)

    logger.info('starting conversion...')
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with Image.open(path.join(idl_dataset_path, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except Exception:
            continue

        camera_id = f'sensor{n}'
        images[(n, camera_id)] = path_secure(filename)  # don't forget windows
        model = kapture.CameraType.UNKNOWN_CAMERA
        if gt_path is not None:
            # replace image extension with .camera
            file_gt_path = os.path.splitext(os.path.join(gt_path, filename))[0] + ".camera"

            if os.path.isfile(file_gt_path):
                with open(file_gt_path) as fin:
                    lines = fin.readlines()
                    lines = (line.rstrip().split() for line in lines)  # split fields
                    lines = list(lines)
                fx = float(lines[0][0])
                cx = float(lines[0][2])
                fy = float(lines[1][1])
                cy = float(lines[1][2])
                width_file = float(lines[8][0])
                height_file = float(lines[8][1])
                assert (width_file == width)
                assert (height_file == height)
                model = kapture.CameraType.PINHOLE
                model_params = [width, height, fx, fy, cx, cy]

                rotation_matrix = [[float(v) for v in line] for line in lines[4:7]]
                rotation = quaternion.from_rotation_matrix(rotation_matrix)
                center_of_projection = [float(v) for v in lines[7]]
                pose = kapture.PoseTransform(rotation, center_of_projection).inverse()
                trajectories[(n, camera_id)] = pose
        cameras[camera_id] = kapture.Camera(model, model_params)

    # if no trajectory were added, no need to create the file
    if not trajectories:
        trajectories = None

    # import (copy) image files.
    logger.info('import image files ...')
    filename_list = [f for _, _, f in kapture.flatten(images)]
    import_record_data_from_dir_auto(idl_dataset_path, kapture_path, filename_list, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_path, imported_kapture)


def import_idl_dataset_cvpr17_command_line() -> None:
    """
    Do the IDL dataset import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='import IDL_dataset_cvpr17 to the kapture format.')
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
    parser.add_argument('-i', '--input', required=True, help='input path to images folder')
    parser.add_argument('-gt', '--gt_path', default=None, help='path to gt data')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_idl_dataset_cvpr17(
        idl_dataset_path=args.input,
        gt_path=args.gt_path,
        kapture_path=args.output,
        force_overwrite_existing=args.force,
        images_import_method=args.image_transfer)


if __name__ == '__main__':
    import_idl_dataset_cvpr17_command_line()
