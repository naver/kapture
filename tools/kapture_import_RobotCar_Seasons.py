#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports a RobotCar dataset to the kapture format.


    The RobotCar Seasons dataset uses a subset of the images provided in the RobotCar
    dataset. It uses images taken under a single reference condition (`overcast-reference`),
    captured at 49 different non-overlapping locations, to represent the scene. For this reference
    condition, the dataset provides a reference 3D model reconstructed using COLMAP [1,2]. The
    3D model consequently defines a set of 6DOF reference poses for the database images.
    In addition, query images taken under different conditions (`dawn`, `dusk`, `night`,
    `night-rain`, `overcast-summer`, `overcast-winter`, `rain`, `snow`, `sun`) at the 49
    locations are provided. For these query images, the reference 6DOF poses will not be
    released.

    T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys,
     J. Sivic, F. Kahl, T. Pajdla.
    Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions.
    Conference on Computer Vision and Pattern Recognition (CVPR) 2018

    The dataset is based on the RobotCar dataset described in this paper:

    W. Maddern, G. Pascoe, C. Linegar, and P. Newman.
    1 Year, 1000km: The Oxford RobotCar Dataset.
    International Journal of Robotics Research (IJRR), 36(1):3–15, 2017

    https://www.visuallocalization.net/datasets/


    Importing:
    - convert all 49 individual colmap models
    - restore rig and timestamps
      - image names: png to jpg
    - convert colmap reference DB to kapture -> needed?
    - merge -> needed?

"""

import argparse
import logging
import numpy as np
import os
import os.path as path
import quaternion
import re
from typing import Dict
# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.converter.colmap.import_colmap import import_colmap

logger = logging.getLogger('RobotCar_Seasons')


def import_robotcar_cameras(intrinsics_dir_path: str) -> kapture.Sensors:
    """
    Read and convert intrinsics files

    :param intrinsics_dir_path:
    :return: kapture.cameras
    """
    cameras = kapture.Sensors()
    for root, dirs, files in os.walk(intrinsics_dir_path):
        for intrinsic_filename in files:
            (camera_id, _) = intrinsic_filename.split('_')
            intrinsic_file = open(path.join(intrinsics_dir_path, intrinsic_filename), 'r')
            (_, fx) = intrinsic_file.readline().split()
            (_, fy) = intrinsic_file.readline().split()
            (_, cx) = intrinsic_file.readline().split()
            (_, cy) = intrinsic_file.readline().split()
            intrinsic_file.close()
            # w, h, fx, fy, cx, cy
            model = kapture.CameraType.PINHOLE
            model_params = [1024, 1024, fx, fy, cx, cy]
            cameras[camera_id] = kapture.Camera(model, model_params)

    return cameras


def import_robotcar_rig(extrinsics_dir_path: str) -> kapture.Rigs:
    """
    Read extrinsics files and convert to kapture rigs

    :param extrinsics_dir_path: path to directory with extrinsics
    :return: kapture.Rigs object
    """
    # From dataset documentation:
    #   The extrinsic parameters of the three cameras on the car with respect to the car itself.
    #   The extrinsics are provided as a 4x4 matrix
    #     [R c]
    #     [0 1]
    #   where R is a rotation matrix that maps from camera to world coordinates and c is the position of the camera in
    #   world coordinates. The entries of the matrix are separated by ,.

    rigs = kapture.Rigs()
    for root, dirs, files in os.walk(extrinsics_dir_path):
        for extrinsics_filepath in files:
            (camera_id, _) = extrinsics_filepath.split('_')
            extrinsic_file = open(path.join(extrinsics_dir_path, extrinsics_filepath), 'r')
            with extrinsic_file as f:
                m = np.array([[float(num) for num in line.split(',')] for line in f])
            rotation_matrix = m[0:3, 0:3]
            position = m[0:3, 3:4]
            pose_transform = kapture.PoseTransform(t=position, r=quaternion.from_rotation_matrix(rotation_matrix))
            # kapture rig format is "rig to sensor"
            rigs['car', camera_id] = pose_transform.inverse()

    return rigs


def import_robotcar_colmap_location(robotcar_path: str,  # noqa: C901: function a bit long but not too complex
                                    colmap_reconstruction_fullpath: path,
                                    kapture_path: str,
                                    rigs: kapture.Rigs,
                                    skip_reconstruction: bool) -> kapture.Kapture:
    """
    Import robotcar data for one location from colmap reconstruction

    :param robotcar_path: path to the robotcar top directory
    :param colmap_reconstruction_fullpath: path to the colmap reconstruction directory
    :param kapture_path: path to the kapture top directory
    :param rigs: kapture rigs to modify
    :param skip_reconstruction: if True, will not add the reconstruction
    :return: a kapture object
    """

    # First, import Colmap reconstruction for given location
    kapture_data = import_colmap(kapture_dir_path=kapture_path,
                                 colmap_reconstruction_dir_path=colmap_reconstruction_fullpath,
                                 colmap_images_dir_path=path.join(robotcar_path, "images"),
                                 skip_reconstruction=skip_reconstruction,
                                 images_import_strategy=TransferAction.skip)  # since filenames are incorrect

    # Post processing:
    # - use correct names for cameras
    # - model was built with PNG files, but we have JPG
    # - recover proper timestamps
    # - recover rig

    # Fix sensors.txt
    camera_mapping = {
        'cam_00001': 'left',
        'cam_00002': 'rear',
        'cam_00003': 'right'
    }
    new_cameras = kapture.Sensors()
    for cam_id in kapture_data.sensors:
        new_cameras[camera_mapping[cam_id]] = kapture_data.sensors[cam_id]
    kapture_data.sensors = new_cameras

    if not skip_reconstruction:
        # Fix keypoints
        # Need to rename .png.kpt to .jpg.kpt files and that's all
        for root, dirs, files in os.walk(kapture_path):
            for file in files:
                if file.endswith('.png.kpt'):
                    os.rename(path.join(root, file), path.join(root, file.replace(".png.kpt", ".jpg.kpt")))

        # observations.txt: png -> jpg
        new_observations = kapture.Observations()
        for point3d_idx in kapture_data.observations:
            for image_path, keypoint_id in kapture_data.observations[point3d_idx]:
                new_observations.add(point3d_idx, image_path.replace(".png", ".jpg"), int(keypoint_id))
        kapture_data.observations = new_observations

    # records_camera.txt
    # timestamps, png->jpg
    new_records_camera = kapture.RecordsCamera()
    records_camera_pattern = re.compile(r'.*/(?P<timestamp>\d+)\.png')
    ts_mapping = {}
    for ts, shot in kapture_data.records_camera.items():
        for cam_id, image_path in shot.items():
            matches = records_camera_pattern.match(image_path)
            if not matches:
                continue
            matches = matches.groupdict()
            timestamp = int(matches['timestamp'])
            ts_mapping[ts] = timestamp
            new_path = image_path.replace(".png", ".jpg")
            new_records_camera[timestamp, camera_mapping[cam_id]] = new_path
    kapture_data.records_camera = new_records_camera

    # trajectories.txt
    new_trajectories = kapture.Trajectories()
    # First recover timestamps and camera names
    for ts, sensor_id in sorted(kapture_data.trajectories.key_pairs()):
        new_trajectories[ts_mapping[ts], camera_mapping[sensor_id]] = kapture_data.trajectories[ts, sensor_id]

    kapture_data.trajectories = new_trajectories
    kapture_data.rigs = rigs

    return kapture_data


def read_robotcar_v2_train(robotcar_path: str) -> Dict[str, kapture.PoseTransform]:
    """
    Read the robot car v2 train data file

    :param robotcar_path: path to the data file
    :return: train data
    """
    with(open(path.join(robotcar_path, 'robotcar_v2_train.txt'), 'r')) as f:
        lines = f.readlines()
        # remove empty lines
        lines = (l2 for l2 in lines if l2.strip())
        # trim trailing EOL
        lines = (l3.rstrip("\n\r") for l3 in lines)
        # split space
        lines = (l4.split() for l4 in lines)
        lines = list(lines)
        train_data = {table[0].replace(".png", ".jpg"):
                      kapture.PoseTransform(r=quaternion.from_rotation_matrix([[float(v) for v in table[1:4]],
                                                                               [float(v) for v in table[5:8]],
                                                                               [float(v) for v in table[9:12]]]),
                                            t=[float(v) for v in [table[4], table[8], table[12]]]).inverse()
                      for table in lines}
    return train_data


def import_robotcar_seasons(robotcar_path: str,  # noqa: C901: function a bit long but not too complex
                            kapture_path: str,
                            force_overwrite_existing: bool = False,
                            images_import_method: TransferAction = TransferAction.skip,
                            import_feature_db: bool = False,
                            skip_reconstruction: bool = False,
                            rig_collapse: bool = False,
                            use_colmap_intrinsics: bool = False,
                            import_v1: bool = False) -> None:
    """
    Read the RobotCar Seasons data, creates several kaptures with training and query data.

    :param robotcar_path: path to the robotcar top directory
    :param kapture_path: path to the kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_method: choose how to import actual image files.
    :param import_feature_db: if True, will import the features from the database
    :param skip_reconstruction: if True, will skip the reconstruction part from the training data
    :param rig_collapse: if True, will collapse the rig
    :param use_colmap_intrinsics: if True, will use the colmap intrinsics
    :param import_v1: if True, will use the version 1 of the format
    """

    os.makedirs(kapture_path, exist_ok=True)

    cameras = import_robotcar_cameras(path.join(robotcar_path, 'intrinsics'))
    rigs = import_robotcar_rig(path.join(robotcar_path, 'extrinsics'))

    logger.info("Importing test data")
    # Test data
    image_pattern = re.compile(r'(?P<condition>.+)/(?P<camera>\w+)/(?P<timestamp>\d+)\.jpg')
    queries_path = path.join(robotcar_path, '3D-models', 'individual', 'queries_per_location')
    kapture_imported_query = {}
    for root, dirs, files in os.walk(queries_path):
        for query_file in files:
            records_camera = kapture.RecordsCamera()
            # Get list of query images
            with open(path.join(queries_path, query_file)) as f:
                for line in f:
                    matches = image_pattern.match(line)
                    image_path = line.strip()
                    if not matches:
                        logger.warning(f"Error matching line in {image_path}")
                        continue
                    matches = matches.groupdict()
                    timestamp = int(matches['timestamp'])
                    camera = str(matches['camera'])
                    # condition = str(matches['condition']) : not used ?
                    records_camera[timestamp, camera] = image_path

                (query_name, _) = query_file.split('.')
                kapture_test = kapture.Kapture(sensors=cameras, rigs=rigs, records_camera=records_camera)
                kapture_imported_query[int(query_name.split('_')[-1])] = kapture_test

    # Training data
    logger.info("Importing training data")
    colmap_reconstructions_path = path.join(robotcar_path, '3D-models', 'individual', 'colmap_reconstructions')
    kapture_imported_training = {}
    for root, dirs, files in os.walk(colmap_reconstructions_path):
        for colmap_reconstruction in dirs:
            (loc_id, _) = colmap_reconstruction.split('_')
            kapture_reconstruction_dir = path.join(kapture_path, f"{int(loc_id):02d}", "mapping")
            delete_existing_kapture_files(kapture_reconstruction_dir, force_overwrite_existing)
            logger.info(f'Converting reconstruction {loc_id} to kapture  ...')
            kapture_reconstruction_data = import_robotcar_colmap_location(
                robotcar_path,
                path.join(colmap_reconstructions_path, colmap_reconstruction),
                kapture_reconstruction_dir,
                rigs,
                skip_reconstruction
            )
            # replace intrinsics with the ones found in the text files
            if not use_colmap_intrinsics:
                kapture_reconstruction_data.sensors = cameras
            kapture_imported_training[int(loc_id)] = kapture_reconstruction_data

    if not import_v1:
        _import_robotcar_v2_train(robotcar_path, kapture_imported_query, kapture_imported_training, image_pattern)

    # apply rig collapse
    if rig_collapse:
        logger.info('replacing camera poses with rig poses.')
        for kapture_mapping in kapture_imported_training.values():
            kapture.rigs_recover_inplace(kapture_mapping.trajectories, rigs, ['rear'])

    # IO operations
    robotcar_image_path = path.join(robotcar_path, "images")
    for loc_id, kapture_query in kapture_imported_query.items():
        loc_id_str = f"{loc_id:02d}"
        logger.info(f'writing test data: {loc_id_str}')
        kapture_test_dir = path.join(kapture_path, loc_id_str, "query")
        delete_existing_kapture_files(kapture_test_dir, force_overwrite_existing)
        if not kapture_query.records_camera:  # all images were removed
            continue
        kapture_to_dir(kapture_test_dir, kapture_query)
        query_images = [f for _, _, f in kapture.flatten(kapture_query.records_camera)]
        import_record_data_from_dir_auto(robotcar_image_path, kapture_test_dir,
                                         query_images, images_import_method)

    for loc_id, kapture_mapping in kapture_imported_training.items():
        loc_id_str = f"{loc_id:02d}"
        logger.info(f'writing mapping data: {loc_id_str}')
        kapture_reconstruction_dir = path.join(kapture_path, f"{loc_id:02d}", "mapping")
        kapture_to_dir(kapture_reconstruction_dir, kapture_mapping)
        mapping_images = [f for _, _, f in kapture.flatten(kapture_mapping.records_camera)]
        import_record_data_from_dir_auto(robotcar_image_path, kapture_reconstruction_dir,
                                         mapping_images, images_import_method)

    if import_feature_db:
        _import_colmap_overcast_reference(robotcar_path, kapture_path, force_overwrite_existing)


def _import_robotcar_v2_train(robotcar_path, kapture_imported_query, kapture_imported_training, image_pattern):
    queries_per_location = {image_name: (ts, cam_id, loc_id)
                            for loc_id, kapture_test in kapture_imported_query.items()
                            for ts, cam_id, image_name in kapture.flatten(kapture_test.records_camera)}
    # read robotcar_v2_train.txt
    v2_train_data = read_robotcar_v2_train(robotcar_path)
    for image_name, pose in v2_train_data.items():
        ts, cam_id, loc_id = queries_per_location[image_name]
        assert cam_id == 'rear'
        kapture_imported_training[loc_id].records_camera[ts, cam_id] = image_name
        kapture_imported_training[loc_id].trajectories[ts, cam_id] = pose
        matches = image_pattern.match(image_name)
        if not matches:
            logger.warning(f"Error matching line in {image_name}")
            continue
        matches = matches.groupdict()
        condition = str(matches['condition'])
        timestamp = str(matches['timestamp'])
        # added left and right images in records_camera
        left_image_name = condition + '/' + 'left' + '/' + timestamp + '.jpg'
        right_image_name = condition + '/' + 'right' + '/' + timestamp + '.jpg'
        kapture_imported_training[loc_id].records_camera[ts, 'left'] = left_image_name
        kapture_imported_training[loc_id].records_camera[ts, 'right'] = right_image_name

        # remove entries from query
        del kapture_imported_query[loc_id].records_camera[ts][cam_id]
        del kapture_imported_query[loc_id].records_camera[ts]['left']
        del kapture_imported_query[loc_id].records_camera[ts]['right']
        del kapture_imported_query[loc_id].records_camera[ts]
    # all remaining query images are kept; reading robotcar_v2_test.txt is not necessary


def _import_colmap_overcast_reference(robotcar_path, kapture_path, force_overwrite_existing):
    # Convert Colmap reference DB to kapture
    kapture_train_dir = path.join(kapture_path, "mapping")
    colmap_db_path = path.join(robotcar_path, "3D-models/overcast-reference.db")
    if path.exists(colmap_db_path):
        delete_existing_kapture_files(kapture_train_dir, force_overwrite_existing)
        kapture_train_data = import_colmap(
            kapture_dir_path=kapture_train_dir,
            colmap_database_filepath=colmap_db_path,
            colmap_reconstruction_dir_path='',
            colmap_images_dir_path=path.join(robotcar_path, "images"),
            no_geometric_filtering=True,
            force_overwrite_existing=force_overwrite_existing,
            images_import_strategy=TransferAction.skip
        )
        logger.info(f'saving feature DB to kapture {kapture_train_dir}  ...')
        kapture_to_dir(kapture_train_dir, kapture_train_data)
    else:
        logger.warning(f'Colmap feature DB {colmap_db_path} does not exist... skipping.')


def import_robotcar_seasons_command_line() -> None:
    """
    Do the RobotCar dataset import to kapture using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description='imports RobotCar Seasons dataset to the kapture format. '
                    'This script will create one kapture dataset'
                    ' for each of the 49 locations, based on the colmap reconstructions')
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
    parser.add_argument('-i', '--input', required=True, help='input path to RobotCar-Seasons root folder')
    parser.add_argument('-o', '--output', required=True, help='kapture output directory')
    parser.add_argument('--import_feature_db', required=False, action='store_true', default=False,
                        help='also convert colmap feature database to kapture format')
    parser.add_argument('--skip_reconstruction', required=False, action='store_true', default=False,
                        help='do not import reconstruction data (3d points, keypoints and observations')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.root_link,
                        help=(f'How to import images [root_link], '
                              f'choose among: {", ".join(a.name for a in TransferAction)}'))
    parser.add_argument('--rig_collapse', action='store_true', default=False,
                        help='Replace camera poses with rig poses.')
    parser.add_argument('--use_colmap_intrinsics', action='store_true', default=False,
                        help=('For mapping images, use intrinsics from the reconstruction (SIMPLE_RADIAL)'
                              ' instead of the intrinsics text files (PINHOLE)'))
    parser.add_argument('--v1', action='store_true', default=False,
                        help='Export RobotCar_Seasons v1 instead of v2.')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_robotcar_seasons(args.input, args.output, args.force, args.image_transfer,
                            args.import_feature_db, args.skip_reconstruction, args.rig_collapse,
                            args.use_colmap_intrinsics, args.v1)


if __name__ == '__main__':
    import_robotcar_seasons_command_line()
