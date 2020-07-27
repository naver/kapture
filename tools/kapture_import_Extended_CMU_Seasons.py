#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports an extended CMU Seasons dataset to the kapture format.
It creates one kapture dataset for each of the 24 slices, based on the txt files

This is an extended version of the dataset published in
T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari,
M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, T. Pajdla.
Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions.

Conference on Computer Vision and Pattern Recognition (CVPR) 2018

This dataset is in turn based on the CMU Visual Localization dataset described here:
Hernan Badino, Daniel Huber, and Takeo Kanade.
The CMU Visual Localization Data Set.
http://3dvis.ri.cmu.edu/data-sets/localization, 2011.

The Extended CMU Seasons dataset uses a subset of the images provided in the CMU Visual Localization dataset.
It uses images taken under a single reference condition (sunny + no foliage),
captured at 17 locations (referred to as slices hereafter), to represent the scene.
For this reference condition, the dataset provides a reference 3D model reconstructed using Structure-from-Motion.
The 3D model consequently defines a set of 6DOF reference poses for the database images.
In addition, query images taken under different conditions at the 17 slices are provided.
Reference poses for around 50% of these images are included in this dataset,
in addition to the reference sequence as described above.

https://www.visuallocalization.net/datasets/

Intrinsics

Please refer to the intrinsics.txt file for a description of how the camera intrinsics are specified.
Note that non-linear distortion are present in the images.
The intrinsics.txt file contains information about this distortion.

Database Lists

For each slice, we provide a text file with the list of database images for that slice.
The text file stores a line per database image. Here is an example from slice 2:
img_00122_c0_1303398475046031us.jpg 0.636779 0.569692 0.379586 -0.354792 -85.392177 55.210379 -2.980700
Here, img_00122 indicates that this image is the 122 image in the dataset.
The infix _c1_ indicates that camera 1 was used to capture the image.
1283347879534213us is the timestamp of the capture time.
The seven numbers after the image name is the camera pose:
the first four are the components of a rotation quaternion, corresponding to a rotation R,
and the last three are the camera center C.
The rotation R corresponds to the first 3x3 subblock of the corresponding camera matrix,
and the camera center C is related to the fourth column t of the camera matrix according to C = -R^T * t,
where R^T denotes the transpose of R.

Query images

A list of all query images for each slice is provided in the test-images-sliceX.txt files.
"""

import logging
import os
import os.path as path
import quaternion
import argparse
import re
import numpy as np
from typing import List, Optional, Tuple

# kapture
import path_to_kapture
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir, float_array_or_none
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.algo.merge_keep_ids import merge_keep_ids

logger = logging.getLogger('Extended_CMU_Seasons')
ECMU_IMAGE_PATTERN = r'img_(?P<image_number>\d+)_(?P<camera_id>\w+)_(?P<timestamp>\d+)us\.jpg'


def import_extended_cmu_seasons_intrinsics(intrinsics_file_path: str) -> kapture.Sensors:
    """
    Read and convert intrinsics file
    Format: [Camera ID] [Distortion model] [image width] [image height] [fx] [fy] [cx] [cy] [k1] [k2] [p1] [p2]

    :param intrinsics_file_path: path to the CMU intrinsics file
    :return: kapture cameras
    """
    cameras = kapture.Sensors()
    with open(intrinsics_file_path) as fin:
        table = fin.readlines()
        # remove comment lines
        table = (l1 for l1 in table if not l1.startswith('#'))
        # remove empty lines
        table = (l2 for l2 in table if l2.strip())
        # trim trailing EOL
        table = (l3.rstrip("\n\r") for l3 in table)
        # split space
        table = (re.split(r'\s+', l4) for l4 in table)
        # remove empty split
        table = ([s for s in l5 if s] for l5 in table)

    for camera_id, distortion_model, *camera_params in table:
        cameras[camera_id] = kapture.Camera(distortion_model, list(camera_params))

    return cameras


def _parse_image_name(image_name: str, compiled_image_pattern) -> Tuple[Optional[int], Optional[str]]:
    matches = compiled_image_pattern.match(image_name)
    if not matches:
        logger.warning(f"Error matching image_name in {image_name}")
        return None, None

    matches = matches.groupdict()

    # image_number = str(matches['image_number'])
    camera_id = str(matches['camera_id'])
    timestamp = int(matches['timestamp'])
    return timestamp, camera_id


def import_extended_cmu_seasons_images(image_list_file_path: str) -> Tuple[kapture.RecordsCamera, kapture.Trajectories]:
    """
    Read image list, name.jpg or name.jpg qw qx qy qz cx cy cz

    :param image_list_file_path: path to the image list file
    :return: kapture images and trajectories
    """

    records_camera = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()

    # name.jpg qw qx qy qz cx cy cz
    # or
    # name.jpg
    with open(image_list_file_path) as fin:
        table = fin.readlines()
        # remove comment lines
        table = (line for line in table if not line.startswith('#'))
        # remove empty lines
        table = (line for line in table if line.strip())
        # trim trailing EOL
        table = (line.rstrip("\n\r") for line in table)
        # split space
        table = (re.split(r'\s+', line) for line in table)
        # remove empty split
        table = ([s for s in line if s] for line in table)

    image_pattern = re.compile(ECMU_IMAGE_PATTERN)
    for line in table:
        image_name = line[0]
        timestamp, camera_id = _parse_image_name(image_name, image_pattern)
        if camera_id is None or timestamp is None:
            continue

        records_camera[(timestamp, camera_id)] = image_name
        if len(line) > 1:  # also contains trajectory
            qw, qx, qy, qz, cx, cy, cz = line[1:]
            quaternion_array = float_array_or_none([qw, qx, qy, qz])
            assert quaternion_array is not None
            center_array = float_array_or_none([cx, cy, cz])
            assert center_array is not None
            rotation = quaternion.from_float_array(quaternion_array)
            # C = -R^T * t -> t = -R * C
            translation = np.matmul(quaternion.as_rotation_matrix(rotation),
                                    -1 * np.array(center_array, dtype=np.float))
            pose = kapture.PoseTransform(r=rotation, t=translation)
            trajectories[(timestamp, camera_id)] = pose

    # if no trajectories were added (query), prefer None
    if not trajectories:
        trajectories = None

    return records_camera, trajectories


def _add_images_from_folder(images_folder_path: str, kapture_data: kapture.Kapture):
    file_list = [os.path.relpath(os.path.join(dirpath, filename), images_folder_path)
                 for dirpath, dirs, filenames in os.walk(images_folder_path)
                 for filename in filenames
                 if filename.endswith('us.jpg')]

    already_added_images = set(image_name for _, _, image_name in kapture.flatten(kapture_data.records_camera))
    image_pattern = re.compile(ECMU_IMAGE_PATTERN)
    for image_name in file_list:
        if image_name not in already_added_images:
            timestamp, camera_id = _parse_image_name(image_name, image_pattern)
            if camera_id is None or timestamp is None:
                continue
            kapture_data.records_camera[(timestamp, camera_id)] = image_name


def import_extended_cmu_seasons(cmu_path: str,
                                top_kaptures_path: str,
                                slice_range: List[int],
                                import_all_files: bool = False,
                                force_overwrite_existing: bool = False,
                                images_import_method: TransferAction = TransferAction.skip) -> None:
    """
    Import extended CMU data to kapture. Will make training and query kaptures for every CMU slice.

    :param cmu_path: path to the top directory of the CMU dataset files
    :param top_kaptures_path: top directory for the kaptures to create
    :param slice_range: range of CMU slices to import
    :param import_all_files: if Tre, will import all files
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    """

    os.makedirs(top_kaptures_path, exist_ok=True)

    cameras = import_extended_cmu_seasons_intrinsics(path.join(cmu_path, 'intrinsics.txt'))

    for slice_n in slice_range:
        # prepare paths
        slice_path = os.path.join(cmu_path, f'slice{slice_n}')
        training_images_path = os.path.join(slice_path, 'database')
        query_images_path = os.path.join(slice_path, 'query')
        gt_trajectories_path = os.path.join(slice_path, f'ground-truth-database-images-slice{slice_n}.txt')
        query_image_list = os.path.join(slice_path, f'test-images-slice{slice_n}.txt')
        query_gt_path = os.path.join(slice_path, 'camera-poses')
        query_gt_list = [os.path.join(query_gt_path, f) for f in os.listdir(query_gt_path)]

        # Import training images
        kapture_training_path = path.join(top_kaptures_path, f'slice{slice_n}', "mapping")
        delete_existing_kapture_files(kapture_training_path, force_overwrite_existing)
        training_records_camera, training_trajectories = import_extended_cmu_seasons_images(gt_trajectories_path)
        training_kapture = kapture.Kapture(sensors=cameras,
                                           records_camera=training_records_camera,
                                           trajectories=training_trajectories)
        if import_all_files:
            _add_images_from_folder(training_images_path, training_kapture)
        kapture_to_dir(kapture_training_path, training_kapture)
        # finally import images
        if images_import_method != TransferAction.skip:
            filename_list = [f for _, _, f in kapture.flatten(training_kapture.records_camera)]
            logger.info(f'importing {len(filename_list)} image files ...')
            import_record_data_from_dir_auto(training_images_path,
                                             kapture_training_path,
                                             filename_list,
                                             images_import_method)
        # Import query images
        kapture_query_path = path.join(top_kaptures_path, f'slice{slice_n}', "query")
        delete_existing_kapture_files(kapture_query_path, force_erase=force_overwrite_existing)
        query_records_camera, query_trajectories = import_extended_cmu_seasons_images(query_image_list)
        query_kapture = kapture.Kapture(sensors=cameras,
                                        records_camera=query_records_camera,
                                        trajectories=query_trajectories)

        # import query gt when possible
        query_gt_kapture = []
        for query_gt_path in query_gt_list:
            query_gt_records_camera, query_gt_trajectories = import_extended_cmu_seasons_images(query_gt_path)
            query_gt_kapture.append(kapture.Kapture(sensors=cameras,
                                                    records_camera=query_gt_records_camera,
                                                    trajectories=query_gt_trajectories))
        data_to_merge = [query_kapture] + query_gt_kapture
        query_kapture = merge_keep_ids(data_to_merge, skip_list=[], data_paths=["" for _ in range(len(data_to_merge))],
                                       kapture_path="", images_import_method=TransferAction.skip)
        if import_all_files:
            _add_images_from_folder(query_images_path, query_kapture)
        kapture_to_dir(kapture_query_path, query_kapture)
        # finally import images
        if images_import_method != TransferAction.skip:
            filename_list = [f for _, _, f in kapture.flatten(query_kapture.records_camera)]
            logger.info(f'importing {len(filename_list)} image files ...')
            import_record_data_from_dir_auto(query_images_path, kapture_query_path, filename_list, images_import_method)


def import_extended_cmu_seasons_command_line() -> None:
    """
    Do the CMU dataset import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='imports Extended CMU Seasons dataset to the kapture format.'
                    'This script will create one kapture dataset'
                    ' for each of the 24 slices, based on the txt files')
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
    parser.add_argument('-i', '--input', required=True, help='input path to Extended CMU root folder')
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.skip,
                        help=f'How to import images [skip], '
                             f'choose among: {", ".join(a.name for a in TransferAction)}')
    parser.add_argument('--slice-range', nargs='+', default=list(range(2, 26)), type=int,
                        help='list of slices to import')
    parser.add_argument('--all-files', action='store_true', default=False,
                        help='Add images present in the folder yet not listed in the txt files.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_extended_cmu_seasons(args.input, args.output,
                                args.slice_range, args.all_files,
                                args.force, args.image_transfer)


if __name__ == '__main__':
    import_extended_cmu_seasons_command_line()
