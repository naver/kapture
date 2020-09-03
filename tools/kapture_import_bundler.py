#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a bundler model into a kapture.
A bundler model file is defined here:

    http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6

    # Bundle file v0.3
    <num_cameras> <num_points>   [two integers]
    <camera1>
    <camera2>
       ...
    <cameraN>
    <point1>
    <point2>
       ...
    <pointM>

    Each camera entry <cameraI> contains the estimated camera intrinsics and extrinsics, and has the form:


        <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
        <R>             [a 3x3 matrix representing the camera rotation]
        <t>             [a 3-vector describing the camera translation]

    The cameras are specified in the order they appear in the list of images.

    Each point entry has the form:


        <position>      [a 3-vector describing the 3D position of the point]
        <color>         [a 3-vector describing the RGB color of the point]
        <view list>     [a list of views the point is visible in]


"""

import argparse
import logging
import os
import os.path as path
import numpy as np
import quaternion
from PIL import Image
# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

logger = logging.getLogger('bundler')
MODEL = kapture.CameraType.RADIAL


def import_bundler(bundler_path: str,
                   image_list_path: str,
                   image_dir_path: str,
                   kapture_dir_path: str,
                   ignore_trajectories: bool,
                   add_reconstruction: bool,
                   force_overwrite_existing: bool = False,
                   images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Imports bundler data and save them as kapture.

    :param bundler_path: path to the bundler model file
    :param image_list_path: path to the file containing the list of image names
    :param image_dir_path: input path to bundler image directory.
    :param kapture_dir_path: path to kapture top directory
    :param ignore_trajectories: if True, will not import the trajectories
    :param add_reconstruction: if True, will create 3D points and observations
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """
    os.makedirs(kapture_dir_path, exist_ok=True)
    delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)

    logger.info('loading all content...')
    # if there is a filter list, parse it
    with open(image_list_path) as file:
        file_content = file.readlines()
    # remove end line char and empty lines
    image_list = [line.rstrip() for line in file_content if line != '\n']

    with open(bundler_path) as file:
        bundler_content = file.readlines()
    # remove end line char and empty lines
    bundler_content = [line.rstrip() for line in bundler_content if line != '\n']
    assert bundler_content[0] == "# Bundle file v0.3"
    # <num_cameras> <num_points>
    line_1 = bundler_content[1].split()
    number_of_cameras = int(line_1[0])
    number_of_points = int(line_1[1])
    offset = 2
    number_of_lines_per_camera = 5  # 1 camera + 3 rotation + 1 translation

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories() if not ignore_trajectories else None
    points3d = [] if add_reconstruction else None
    keypoints = kapture.Keypoints('sift', np.float32, 2) if add_reconstruction else None
    observations = kapture.Observations() if add_reconstruction else None if add_reconstruction else None
    image_mapping = []  # bundler camera_id -> (name, width, height)
    for i in range(0, number_of_cameras):
        start_index = i * number_of_lines_per_camera + offset
        file_name = image_list[i]

        # process camera info
        line_camera = bundler_content[start_index].split()
        focal_length = float(line_camera[0])
        k1 = float(line_camera[1])
        k2 = float(line_camera[2])

        # lazy open
        with Image.open(path.join(image_dir_path, file_name)) as im:
            width, height = im.size

        image_mapping.append((file_name, width, height))
        camera = kapture.Camera(MODEL, [width, height, focal_length, width / 2, height / 2, k1, k2])
        camera_id = f'sensor{i}'
        cameras[camera_id] = camera

        # process extrinsics
        rotation_matrix = [[float(v) for v in line.split()]
                           for line in bundler_content[start_index + 1:start_index + 4]]

        quaternion_wxyz = quaternion.from_rotation_matrix(rotation_matrix)
        translation = np.array([float(v) for v in bundler_content[start_index + 4].split()])
        pose = kapture.PoseTransform(quaternion_wxyz, translation)

        # The Bundler model uses a coordinate system that differs from the *computer vision camera
        #  coordinate system*. More specifically, they use the camera coordinate system typically used
        #  in *computer graphics*. In this camera coordinate system, the camera is looking down the
        #  `-z`-axis, with the `x`-axis pointing to the right and the `y`-axis pointing upwards.
        # rotation Pi around the x axis to get the *computer vision camera
        #  coordinate system*
        rotation_around_x = quaternion.quaternion(0.0, 1.0, 0.0, 0.0)
        transformation = kapture.PoseTransform(rotation_around_x, np.array([0, 0, 0]))

        images[(i, camera_id)] = file_name
        if trajectories is not None:
            # transformation.inverse() is equal to transformation (rotation around -Pi or Pi around X is the same)
            trajectories[(i, camera_id)] = kapture.PoseTransform.compose([transformation, pose, transformation])

    if points3d is not None and number_of_points > 0:
        assert keypoints is not None
        assert observations is not None
        offset += number_of_cameras * number_of_lines_per_camera
        number_of_lines_per_point = 3  # position color viewlist

        # (image_name, bundler_keypoint_id ) -> keypoint_id
        known_keypoints = {}
        local_keypoints = {}

        for i in range(0, number_of_points):
            start_index = i * number_of_lines_per_point + offset
            position = [float(v) for v in bundler_content[start_index].split()]
            # apply transformation
            position = [position[0], -position[1], -position[2]]
            color = [float(v) for v in bundler_content[start_index + 1].split()]

            # <view list>: length of the list + [<camera> <key> <x> <y>]
            # x, y origin is the center of the image
            view_list = bundler_content[start_index + 2].split()
            number_of_observations = int(view_list[0])

            for j in range(number_of_observations):
                camera_id = int(view_list[1 + 4 * j + 0])
                keypoint_id = int(view_list[1 + 4 * j + 1])
                x = float(view_list[1 + 4 * j + 2])
                y = float(view_list[1 + 4 * j + 2])

                file_name, width, height = image_mapping[camera_id]
                # put (0,0) in upper left corner
                x += (width / 2)
                y += (height / 2)

                # init local_keypoints if needed
                if file_name not in local_keypoints:
                    local_keypoints[file_name] = []
                # do not add the same keypoint twice
                if (file_name, keypoint_id) not in known_keypoints:
                    # in the kapture format, keypoint id is different. Note that it starts from 0
                    known_keypoints[(file_name, keypoint_id)] = len(local_keypoints[file_name])
                    local_keypoints[file_name].append([x, y])
                keypoint_idx = known_keypoints[(file_name, keypoint_id)]
                observations.add(i, file_name, keypoint_idx)
            points3d.append(position + color)
        points3d = np.array(points3d)

        # finally, convert local_keypoints to np.ndarray and add them to the global keypoints variable
        keypoints = kapture.Keypoints('sift', np.float32, 2)
        for image_filename, keypoints_array in local_keypoints.items():
            keypoints_np_array = np.array(keypoints_array)
            keypoints_out_path = kapture.io.features.get_keypoints_fullpath(kapture_dir_path, image_filename)
            kapture.io.features.image_keypoints_to_file(keypoints_out_path, keypoints_np_array)
            keypoints.add(image_filename)

    if points3d is not None:
        points3d = kapture.Points3d(points3d)

    # import (copy) image files.
    logger.info('import image files ...')
    filename_list = [f for _, _, f in kapture.flatten(images)]
    import_record_data_from_dir_auto(image_dir_path, kapture_dir_path, filename_list, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories,
                                       points3d=points3d, keypoints=keypoints, observations=observations)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_dir_path, imported_kapture)


def bundler_command_line() -> None:
    """
    Do the bundler import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='import bundler file to the kapture format.')
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
                        help='input path to .out bundler file')
    parser.add_argument('-l', '--image-list', required=True,
                        help='path to the list of images in the bundler order (.list.txt)')
    parser.add_argument('-im', '--image-path', required=True, help='path to images')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    parser.add_argument('--ignore-trajectories', action='store_true', default=False,
                        help='Do not export extrinsics.')
    parser.add_argument('--add-reconstruction', action='store_true', default=False,
                        help='add the 3d points/keypoints/observations to the output')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_bundler(bundler_path=args.input,
                   image_list_path=args.image_list,
                   image_dir_path=args.image_path,
                   kapture_dir_path=args.output,
                   ignore_trajectories=args.ignore_trajectories,
                   add_reconstruction=args.add_reconstruction,
                   force_overwrite_existing=args.force,
                   images_import_method=args.image_transfer)


if __name__ == '__main__':
    bundler_command_line()
