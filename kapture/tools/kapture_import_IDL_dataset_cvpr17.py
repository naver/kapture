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
from PIL import Image
import quaternion
from typing import Optional

# kapture
import kapture
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.utils.paths import path_secure
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


logger = logging.getLogger('IDL_dataset_cvpr17')


def import_idl_dataset_cvpr17(input: str,
                              output: str,
                              gt_path: Optional[str] = None,
                              force: bool = False,
                              image_transfer: TransferAction = TransferAction.link_absolute,
                              verbose: Verbosity = DEFAULT_VERBOSITY,
                              quiet: bool = False) -> None:
    """
    Imports IDL_dataset_cvpr17 to the kapture format.

    :param input: path to the IDL dataset images folder
    :param output: path to the kapture top directory to create
    :param gt_path: path to ground truth data
    :param force: silently overwrite kapture files if already exists.
    :param image_transfer: how to import actual image files.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)

    os.makedirs(output, exist_ok=True)
    delete_existing_kapture_files(output, force_erase=force)

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()

    file_list = [os.path.relpath(os.path.join(dirpath, filename), input)
                 for dirpath, dirs, filenames in os.walk(input)
                 for filename in filenames]
    file_list = sorted(file_list)

    logger.info('starting conversion...')
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with Image.open(path.join(input, filename)) as im:
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
    import_record_data_from_dir_auto(input, output, filename_list, image_transfer)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories)
    logger.info('writing imported data...')
    kapture_to_dir(output, imported_kapture)


def main():
    run_cli(import_idl_dataset_cvpr17, aliases={'--gt_path': ['-gt']})


if __name__ == '__main__':
    main()
