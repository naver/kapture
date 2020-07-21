#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports an NVM model in the kapture format.

VisualSFM saves SfM workspaces into NVM files, which contain input image paths and multiple 3D models.
Below is the format description

NVM_V3 [optional calibration]                        # file version header
<Model1> <Model2> ...                                # multiple reconstructed models
<Empty Model containing the unregistered Images>     # number of camera > 0, but number of points = 0
<0>                                                  # 0 camera to indicate the end of model section
<Some comments describing the PLY section>
<Number of PLY files> <List of indices of models that have associated PLY>

The [optional calibration] exists only if you use "Set Fixed Calibration" Function
FixedK fx cx fy cy

Each reconstructed <model> contains the following
<Number of cameras> <List of cameras>
<Number of 3D points> <List of points>

The cameras and 3D points are saved in the following format
<Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
<Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
<Measurement> = <Image index> <Feature Index> <xy>
"""

import logging
import os
import os.path as path
import PIL
from PIL import Image
import numpy as np
import quaternion
import argparse
from typing import List, Optional, Set

# kapture
import path_to_kapture
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
import kapture.io.features


logger = logging.getLogger('nvm')
MODEL = kapture.CameraType.SIMPLE_RADIAL


def import_nvm(nvm_file_path: str,
               nvm_images_path: str,
               kapture_path: str,
               filter_list_path: Optional[str],
               ignore_trajectories: bool,
               add_reconstruction: bool,
               force_overwrite_existing: bool = False,
               images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Imports nvm data to kapture format.

    :param nvm_file_path: path to nvm file
    :param nvm_images_path: path to NVM images directory.
    :param kapture_path: path to kapture root directory.
    :param filter_list_path: path to the optional file containing a list of images to process
    :param ignore_trajectories: if True, will not create trajectories
    :param add_reconstruction: if True, will add observations, keypoints and 3D points.
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """

    # TODO implement [optional calibration]
    # doc : http://ccwu.me/vsfm/doc.html#nvm
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_erase=force_overwrite_existing)

    logger.info('loading all content...')
    # if there is a filter list, parse it
    # keep it as Set[str] to easily find images
    if filter_list_path:
        with open(filter_list_path) as file:
            file_content = file.readlines()
        # remove end line char and empty lines
        filter_list = {line.rstrip() for line in file_content if line != '\n'}
    else:
        filter_list = None

    # now do the nvm
    with open(nvm_file_path) as file:
        nvm_content = file.readlines()
    # remove end line char and empty lines
    nvm_content = [line.rstrip() for line in nvm_content if line != '\n']
    # only NVM_V3 is supported
    assert nvm_content[0] == "NVM_V3"
    # offset represents the line pointer
    offset = 1
    # camera_id_offset keeps tracks of used camera_id in case of multiple reconstructed models
    camera_id_offset = 0
    # point_id_offset keeps tracks of used point_id in case of multiple reconstructed models
    point_id_offset = 0

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories() if not ignore_trajectories else None
    observations = kapture.Observations() if add_reconstruction else None if add_reconstruction else None
    keypoints = kapture.Keypoints('sift', np.float32, 2) if add_reconstruction else None
    points3d = [] if add_reconstruction else None

    # break if number of cameras == 0 or reached end of file
    while True:
        # <Model1> <Model2> ...
        # Each reconstructed <model> contains the following
        # <Number of cameras> <List of cameras>
        # <Number of 3D points> <List of points>
        # In practice,
        # <Number of cameras>
        # <List of cameras>, one per line
        # <Number of 3D points>
        # <List of points>, one per line
        number_of_cameras = int(nvm_content[offset])
        offset += 1
        if number_of_cameras == 0:  # a line with <0> signify the end of models
            break

        logger.debug('importing model cameras...')
        # parse all cameras for current model
        image_idx_to_image_name = parse_cameras(number_of_cameras,
                                                nvm_content,
                                                offset,
                                                camera_id_offset,
                                                filter_list,
                                                nvm_images_path,
                                                cameras,
                                                images,
                                                trajectories)
        offset += number_of_cameras
        camera_id_offset += number_of_cameras

        # parse all points3d
        number_of_points = int(nvm_content[offset])
        offset += 1
        if points3d is not None and number_of_points > 0:
            assert keypoints is not None
            assert observations is not None
            logger.debug('importing model points...')
            parse_points3d(kapture_path,
                           number_of_points,
                           nvm_content,
                           offset,
                           point_id_offset,
                           image_idx_to_image_name,
                           filter_list,
                           points3d,
                           keypoints,
                           observations)

        point_id_offset += number_of_points
        offset += number_of_points
        # reached end of file?
        if offset >= len(nvm_content):
            break

    # do not export values if none were found.
    if points3d is not None:
        points3d = kapture.Points3d(points3d)

    # import (copy) image files.
    logger.info('import image files ...')
    images_filenames = [f for _, _, f in kapture.flatten(images)]
    import_record_data_from_dir_auto(nvm_images_path, kapture_path, images_filenames, images_import_method)

    # pack into kapture format
    imported_kapture = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories,
                                       points3d=points3d, keypoints=keypoints, observations=observations)
    logger.info('writing imported data...')
    kapture_to_dir(kapture_path, imported_kapture)


def parse_cameras(number_of_cameras: int,
                  nvm_content: List[str],
                  offset: int,
                  camera_id_offset: int,
                  filter_list: Optional[Set[str]],
                  nvm_images_path: str,
                  cameras: kapture.Sensors,
                  images: kapture.RecordsCamera,
                  trajectories: Optional[kapture.Trajectories]) -> List[str]:
    """
    Parse the <List of cameras> section
    Fill cameras, images, trajectories in place.
    Image files must exist to be able to retrieve height and width.

    :param number_of_cameras: number of cameras to process
    :param nvm_content: content of NVM file
    :param offset: number of characters to skip while reading every line
    :param camera_id_offset:
    :param filter_list: optional list of images to process
    :param nvm_images_path: path to NVM images directory
    :param cameras: kapture cameras to extend
    :param images: kapture images to extend
    :param trajectories: kapture trajectories to extend
    :return: list of images with position = index
    """
    image_idx_to_image_name = []
    # parse all cameras
    for i in range(0, number_of_cameras):
        line = nvm_content[i + offset].split()
        timestamp = i + camera_id_offset
        camera_id = f'sensor{timestamp}'
        image_file_name = line[0]
        image_idx_to_image_name.append(image_file_name)
        if filter_list is not None and image_file_name not in filter_list:
            # file_name is not in the list, do not add it
            continue

        focal_length = float(line[1])
        quaternion_wxyz = quaternion.from_float_array([float(v) for v in line[2:6]])
        camera_center = np.array([float(v) for v in line[6:9]])
        # https://github.com/colmap/colmap/blob/67e96894d4beed7cc93f1c0755a98d3664f85e63/src/base/reconstruction.cc#L891
        radial_distortion = -float(line[9])  # SIGN !

        try:
            # lazy open
            with Image.open(path.join(nvm_images_path, image_file_name)) as im:
                width, height = im.size
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            logger.info(f'Skipping invalid image file {image_file_name}')
            continue

        translation = - np.matmul(quaternion.as_rotation_matrix(quaternion_wxyz), camera_center)
        pose = kapture.PoseTransform(quaternion_wxyz, translation)

        camera = kapture.Camera(MODEL, [width, height, focal_length, width / 2, height / 2, radial_distortion])
        cameras[camera_id] = camera

        images[(timestamp, camera_id)] = image_file_name
        if trajectories is not None:
            trajectories[(timestamp, camera_id)] = pose
    return image_idx_to_image_name


def parse_points3d(kapture_path: str,
                   number_of_points: int,
                   nvm_content: List[str],
                   offset: int,
                   point_id_offset: int,
                   image_idx_to_image_name: List[str],
                   filter_list: Optional[Set[str]],
                   points3d: List[List[float]],
                   keypoints: kapture.Keypoints,
                   observations: kapture.Observations) -> None:
    """
    Parse the <List of points> section
    Fill points3d, keypoints, observations in place
    Write keypoints to disk.

    :param kapture_path: path to kapture root directory.
    :param number_of_points: number of points to process
    :param nvm_content: content of NVM file
    :param offset: number of characters to skip while reading every line
    :param point_id_offset:
    :param image_idx_to_image_name: list of images in their index order
    :param filter_list: optional list of images to process
    :param points3d: list of 3D points to extend
    :param keypoints: kapture keypoints list to extend
    :param observations: kapture observations to extend
    """
    # (image_name, nvm_feature_id ) -> keypoint_id
    known_keypoints = {}
    local_keypoints = {}
    for i in range(0, number_of_points):
        fields = nvm_content[i + offset].split()
        points3d.append([float(v) for v in fields[0:6]])
        # parse observations
        number_of_measurements = int(fields[6])
        for j in range(0, number_of_measurements):
            # parse measurement
            image_index = int(fields[7 + 4 * j + 0])
            feature_index = int(fields[7 + 4 * j + 1])
            x = float(fields[7 + 4 * j + 2])
            y = float(fields[7 + 4 * j + 3])

            # retrieve filename. if added, then proceed to add features / observations
            file_name = image_idx_to_image_name[image_index]
            if filter_list is not None and file_name not in filter_list:
                # file_name is not in the list, do not add it
                continue

            # init local_keypoints if needed
            if file_name not in local_keypoints:
                local_keypoints[file_name] = []
            # do not add the same keypoint twice
            if (file_name, feature_index) not in known_keypoints:
                # in the kapture format, keypoint id is different. Note that it starts from 0
                known_keypoints[(file_name, feature_index)] = len(local_keypoints[file_name])
                local_keypoints[file_name].append([x, y])
            keypoint_idx = known_keypoints[(file_name, feature_index)]
            point3d_idx = i + point_id_offset
            observations.add(point3d_idx, file_name, keypoint_idx)

    # finally, convert local_keypoints to np.ndarray and add them to the global keypoints variable
    for image_filename, keypoints_array in local_keypoints.items():
        keypoints_np_array = np.array(keypoints_array)
        keypoints_filepath = kapture.io.features.get_keypoints_fullpath(kapture_path, image_filename)
        kapture.io.features.image_keypoints_to_file(keypoints_filepath, keypoints_np_array)
        keypoints.add(image_filename)


def import_nvm_command_line() -> None:
    """
    Do the NVM import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='import nvm file to the kapture format.')
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
    parser.add_argument('-i', '--input', required=True, help='input path to nvm file')
    parser.add_argument('-im', '--images', default=None,
                        help='path to images directory.')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.link_absolute,
                        help=f'How to import images [link_absolute], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    parser.add_argument('--filter-list', default="",
                        help=('path to the filter list (optional), '
                              'this file contains a list a images, '
                              'one per line that will be kept in the output. '
                              'images not in the list will be skipped'))
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

    import_nvm(args.input, args.images, args.output, args.filter_list,
               args.ignore_trajectories, args.add_reconstruction, args.force, args.image_transfer)


if __name__ == '__main__':
    import_nvm_command_line()
