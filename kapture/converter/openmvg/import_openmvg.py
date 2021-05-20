# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Openmvg to kapture import functions.
"""

import json
import logging
import quaternion
import numpy as np
import os
import os.path as path
import shutil
from tqdm import tqdm
from typing import Dict, Optional, Union
# kapture
import kapture
import kapture.io.csv as kcsv
import kapture.io.structure
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.io.features import get_keypoints_fullpath, get_descriptors_fullpath
from kapture.io.features import get_matches_fullpath
from kapture.io.binary import array_to_file
from kapture.utils.paths import path_secure
# local
from .openmvg_commons import JSON_KEY, OPENMVG_DEFAULT_JSON_FILE_NAME
from .openmvg_commons import CameraModel

logger = logging.getLogger('openmvg')  # Using global openmvg logger


def import_openmvg(
        sfm_data_path: str,
        regions_dir_path: Optional[str],
        matches_file_path: Optional[str],
        kapture_path: str,
        image_action: TransferAction,
        force_overwrite_existing: bool = False) -> None:
    """
    Converts an openMVG JSON file to a kapture directory.
    If an image action is provided (link, copy or move), links to the image files are created,
    or the image files are copied or moved.

    :param sfm_data_path: path to the openMVG sfm_data file.
    :param regions_dir_path: input path to directory containing regions (*.feat, *.desc)
    :param matches_file_path: input path to openMVG matches file (eg. matches.f.bin)
    :param kapture_path: path to the kapture directory where the data will be exported
    :param image_action: action to apply to the images
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    """

    if path.isdir(sfm_data_path):
        sfm_data_path = path.join(sfm_data_path, OPENMVG_DEFAULT_JSON_FILE_NAME)
    # sanity check
    if not path.isfile(sfm_data_path):
        raise ValueError(f'OpenMVG JSON file {sfm_data_path} does not exist.')

    os.makedirs(kapture_path, exist_ok=True)
    kapture.io.structure.delete_existing_kapture_files(kapture_path, force_overwrite_existing)

    logger.info(f'Loading sfm_data file {sfm_data_path}')
    with open(sfm_data_path, 'r') as f:
        input_json = json.load(f)
        kapture_data = import_openmvg_sfm_data_json(input_json, kapture_path, image_action)

    if regions_dir_path:
        logger.info(f'Loading regions from {regions_dir_path}')
        _import_openmvg_regions(regions_dir_path, kapture_data, kapture_path)

    if matches_file_path:
        logger.info(f'Loading matches from {matches_file_path}')
        _import_openmvg_matches(matches_file_path, kapture_data, kapture_path)

    logger.info(f'Saving to kapture {kapture_path}')
    kcsv.kapture_to_dir(kapture_path, kapture_data)


GET_ID_MASK = 2147483647  # 01111111 11111111 11111111 11111111
ID_POSE_NOT_LOCALIZED = 4294967295  # 11111111 11111111 11111111 11111111


def import_openmvg_sfm_data_json(
        sfm_data_json: Dict[str, Union[str, Dict]],
        kapture_images_path: str,
        image_action=TransferAction.skip) -> kapture.Kapture:
    """
    Imports an openMVG sfm_data json structure to a kapture object.
    Also copy, move or link the images files if necessary.

    :param sfm_data_json: the openmvg JSON parsed as a dictionary
    :param kapture_images_path: top directory to create the kapture images tree
    :param image_action: action to apply on images: link, copy, move or do nothing.
    :return: the constructed kapture object
    """

    data_root_path: str = ''

    if sfm_data_json[JSON_KEY.ROOT_PATH]:
        data_root_path = sfm_data_json[JSON_KEY.ROOT_PATH]
    elif image_action == TransferAction.skip:
        logger.warning('No root_path in sfm_data.')
    else:  # It is needed to execute an action with the image file
        raise ValueError(f"Missing root_path to do image action '{image_action.name}'")
    openmvg_images_dir = path.basename(data_root_path)

    # Imports all the data from the json file to kapture objects
    kapture_cameras = _import_openmvg_cameras(sfm_data_json)
    device_identifiers = {int: str}  # Pose id -> device id
    timestamp_for_pose = {int: int}  # Pose id -> timestamp
    # Imports the images as records_camera, but also fill in the devices_identifiers and timestamp_for_pose dictionaries
    records_camera = _import_openmvg_images(
        sfm_data_json, image_action, kapture_images_path, openmvg_images_dir, data_root_path,
        device_identifiers, timestamp_for_pose)
    trajectories = _import_openmvg_trajectories(
        sfm_data_json, device_identifiers, timestamp_for_pose)

    kapture_data = kapture.Kapture(sensors=kapture_cameras, records_camera=records_camera, trajectories=trajectories)
    return kapture_data


def _import_openmvg_cameras(input_json) -> kapture.Sensors:  # noqa: C901
    kapture_cameras = kapture.Sensors()
    if input_json.get(JSON_KEY.INTRINSICS):
        polymorphic_id_to_value = {}
        logger.info('Importing intrinsics')
        for sensor in input_json[JSON_KEY.INTRINSICS]:
            value = sensor[JSON_KEY.VALUE]
            if JSON_KEY.POLYMORPHIC_NAME in value:
                # new type name: store it for next instances
                polymorphic_id = value[JSON_KEY.POLYMORPHIC_ID] & GET_ID_MASK
                polymorphic_id_to_value[polymorphic_id] = value[JSON_KEY.POLYMORPHIC_NAME]
                logger.debug("New camera_type: " + polymorphic_id_to_value[polymorphic_id])
            else:
                if JSON_KEY.POLYMORPHIC_ID not in value:
                    raise ValueError(f'{JSON_KEY.POLYMORPHIC_ID} is missing (intrinsics)')
                polymorphic_id = value[JSON_KEY.POLYMORPHIC_ID]

            if polymorphic_id not in polymorphic_id_to_value:
                raise ValueError(f'Unknown polymorphic_id {polymorphic_id}')

            camera_model = CameraModel(polymorphic_id_to_value[polymorphic_id])
            camera_data = value[JSON_KEY.PTR_WRAPPER][JSON_KEY.DATA]

            if camera_model == CameraModel.pinhole:
                # w, h, f, cx, cy
                camera = kapture.Camera(kapture.CameraType.SIMPLE_PINHOLE, [
                    int(camera_data[JSON_KEY.WIDTH]),
                    int(camera_data[JSON_KEY.HEIGHT]),
                    camera_data[JSON_KEY.FOCAL_LENGTH],
                    camera_data[JSON_KEY.PRINCIPAL_POINT][0],
                    camera_data[JSON_KEY.PRINCIPAL_POINT][1],
                ])
            elif camera_model == CameraModel.pinhole_radial_k1:
                # w, h, f, cx, cy, k
                camera = kapture.Camera(kapture.CameraType.SIMPLE_RADIAL, [
                    int(camera_data[JSON_KEY.WIDTH]),
                    int(camera_data[JSON_KEY.HEIGHT]),
                    camera_data[JSON_KEY.FOCAL_LENGTH],
                    camera_data[JSON_KEY.PRINCIPAL_POINT][0],
                    camera_data[JSON_KEY.PRINCIPAL_POINT][1],
                    camera_data[JSON_KEY.DISTO_K1][0]
                ])
            elif camera_model == CameraModel.pinhole_radial_k3:
                # w, h, f, cx, cy, k1, k2, k3
                camera = kapture.Camera(kapture.CameraType.RADIAL, [
                    int(camera_data[JSON_KEY.WIDTH]),
                    int(camera_data[JSON_KEY.HEIGHT]),
                    camera_data[JSON_KEY.FOCAL_LENGTH],
                    camera_data[JSON_KEY.PRINCIPAL_POINT][0],
                    camera_data[JSON_KEY.PRINCIPAL_POINT][1],
                    camera_data[JSON_KEY.DISTO_K3][0],
                    camera_data[JSON_KEY.DISTO_K3][1]
                ])
                # camera_data["disto_k3"][2] ignored: radial model has two distortion param, while openMVG's has three
            elif camera_model == CameraModel.pinhole_brown_t2:
                # w, h, f, cx, cy, k1, k2, k3, t1, t2
                if float(camera_data[JSON_KEY.DISTO_T2][2]) != 0:
                    # if k3 not null, use FULL_OPENCV, otherwise OPENCV
                    # w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                    value0 = camera_data[JSON_KEY.VALUE0]
                    disto_t2 = camera_data[JSON_KEY.DISTO_T2]
                    camera = kapture.Camera(kapture.CameraType.FULL_OPENCV, [
                        int(value0[JSON_KEY.WIDTH]),
                        int(value0[JSON_KEY.HEIGHT]),
                        value0[JSON_KEY.FOCAL_LENGTH],
                        value0[JSON_KEY.FOCAL_LENGTH],
                        value0[JSON_KEY.PRINCIPAL_POINT][0],
                        value0[JSON_KEY.PRINCIPAL_POINT][1],
                        disto_t2[0], disto_t2[1], disto_t2[3], disto_t2[4], disto_t2[2],
                        0, 0, 0
                    ])
                else:
                    # w, h, fx, fy, cx, cy, k1, k2, p1, p2
                    value0 = camera_data[JSON_KEY.VALUE0]
                    disto_t2 = camera_data[JSON_KEY.DISTO_T2]
                    camera = kapture.Camera(kapture.CameraType.OPENCV, [
                        int(value0[JSON_KEY.WIDTH]),
                        int(value0[JSON_KEY.HEIGHT]),
                        value0[JSON_KEY.FOCAL_LENGTH],
                        value0[JSON_KEY.FOCAL_LENGTH],
                        value0[JSON_KEY.PRINCIPAL_POINT][0],
                        value0[JSON_KEY.PRINCIPAL_POINT][1],
                        disto_t2[0], disto_t2[1], disto_t2[3], disto_t2[4]])
            elif camera_model == CameraModel.fisheye:
                logger.warning(
                    "OpenMVG fisheye models are not compatible with OpenCV."
                    " Using SIMPLE_RADIAL_FISHEYE and forcing distortion to 0")
                # w, h, f, cx, cy, k
                value0 = camera_data[JSON_KEY.VALUE0]
                camera = kapture.Camera(kapture.CameraType.SIMPLE_RADIAL_FISHEYE, [
                    int(value0[JSON_KEY.WIDTH]),
                    int(value0[JSON_KEY.HEIGHT]),
                    value0[JSON_KEY.FOCAL_LENGTH],
                    value0[JSON_KEY.PRINCIPAL_POINT][0],
                    value0[JSON_KEY.PRINCIPAL_POINT][1],
                    0])
            else:
                raise ValueError(f'Camera model {camera_model} not supported')

            kapture_cameras[str(sensor[JSON_KEY.KEY])] = camera

    return kapture_cameras


def _import_openmvg_images(input_json, image_action, kapture_images_path, openmvg_images_dir, root_path,
                           device_identifiers, timestamp_for_pose):
    records_camera = kapture.RecordsCamera()
    if input_json.get(JSON_KEY.VIEWS):
        views = input_json[JSON_KEY.VIEWS]
        if image_action == TransferAction.root_link:
            # Do a unique images directory link
            # kapture/<records_dir>/openmvg_top_images_directory -> openmvg_root_path
            kapture_records_path = get_image_fullpath(kapture_images_path)
            os.makedirs(kapture_records_path, exist_ok=True)
            os.symlink(root_path, path.join(kapture_records_path, openmvg_images_dir))
        logger.info(f'Importing {len(views)} images')
        # Progress bar only in debug or info level
        if image_action != TransferAction.skip and image_action != TransferAction.root_link \
                and logger.getEffectiveLevel() <= logging.INFO:
            progress_bar = tqdm(total=len(views))
        else:
            progress_bar = None
        for view in views:
            input_data = view[JSON_KEY.VALUE][JSON_KEY.PTR_WRAPPER][JSON_KEY.DATA]
            pose_id = input_data[JSON_KEY.ID_POSE]
            # All two values should be the same (?)
            if input_data[JSON_KEY.ID_VIEW]:
                timestamp = input_data[JSON_KEY.ID_VIEW]
            else:
                timestamp = view[JSON_KEY.KEY]
            device_id = str(input_data[JSON_KEY.ID_INTRINSIC])  # device_id must be a string for kapture
            device_identifiers[pose_id] = device_id
            timestamp_for_pose[pose_id] = timestamp

            kapture_filename = _import_openmvg_image_file(input_data, openmvg_images_dir, root_path,
                                                          kapture_images_path, image_action)

            progress_bar and progress_bar.update(1)

            key = (timestamp, device_id)  # tuple of int,str
            records_camera[key] = path_secure(kapture_filename)
        progress_bar and progress_bar.close()
    return records_camera


def _import_openmvg_image_file(input_data, openmvg_images_dir, root_path, kapture_images_path, image_action) -> str:
    # Add the common openmvg images directory in front of the filename
    filename: str
    if input_data.get(JSON_KEY.LOCAL_PATH):
        filename = path.join(input_data[JSON_KEY.LOCAL_PATH], input_data[JSON_KEY.FILENAME])
    else:
        filename = input_data[JSON_KEY.FILENAME]
    kapture_filename = path.join(openmvg_images_dir, filename)
    if image_action != TransferAction.skip and image_action != TransferAction.root_link:
        src_path: str
        if root_path:
            src_path = path.join(root_path, filename)
        else:
            src_path = filename
        dst_path = get_image_fullpath(kapture_images_path, kapture_filename)
        # Create destination directory if necessary
        dst_dir = path.dirname(dst_path)
        if not path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        # Check if already exist
        if path.exists(dst_path):
            os.unlink(dst_path)
        # Create file or link
        if image_action == TransferAction.copy:
            shutil.copy2(src_path, dst_path)
        elif image_action == TransferAction.move:
            shutil.move(src_path, dst_path)
        else:
            # Individual link
            if image_action == TransferAction.link_relative:
                # Compute relative path
                src_path = path.relpath(src_path, dst_dir)
            os.symlink(src_path, dst_path)
            # Symlink might crash on Windows if the user executing this code has no admin privilege
    return kapture_filename


def _import_openmvg_trajectories(input_json, device_identifiers, timestamp_for_pose):
    trajectories = kapture.Trajectories()
    if input_json.get(JSON_KEY.EXTRINSICS):
        extrinsics = input_json[JSON_KEY.EXTRINSICS]
        logger.info(f'Importing {len(extrinsics)} extrinsics -> trajectories')
        for pose in extrinsics:
            pose_id = pose[JSON_KEY.KEY]
            center = pose[JSON_KEY.VALUE][JSON_KEY.CENTER]
            rotation = pose[JSON_KEY.VALUE][JSON_KEY.ROTATION]
            kap_translation = -1 * np.matmul(rotation, center)
            kap_pose = kapture.PoseTransform(quaternion.from_rotation_matrix(rotation), kap_translation)
            timestamp = timestamp_for_pose.get(pose_id)
            if timestamp is None:
                logger.warning(f'Missing timestamp for extrinsic {pose_id}')
                continue
            device_id = device_identifiers.get(pose_id)
            if device_id is None:
                logger.warning(f'Missing device for extrinsic {pose_id}')
                continue
            trajectories[(timestamp, device_id)] = kap_pose  # tuple of int,str -> 6D pose
    return trajectories


def _import_openmvg_regions(
        openmvg_regions_directory_path,
        kapture_data,
        kapture_path
):
    # look for the "image_describer.json"
    image_describer_path = path.join(openmvg_regions_directory_path, 'image_describer.json')
    if not path.isfile(image_describer_path):
        logger.debug(f'file not found : {image_describer_path}')
        return

    with open(image_describer_path) as f:
        image_describer = json.load(f)

    # retrieve what type of keypoints it is.
    keypoints_type = image_describer.get('regions_type', {}).get('polymorphic_name', 'UNDEFINED')
    keypoints_name = {
        'SIFT_Regions': 'SIFT',
        'AKAZE_Float_Regions': 'AKAZE'
    }.get(keypoints_type, keypoints_type)
    kapture_keypoints = kapture.Keypoints(type_name=keypoints_name, dtype=float, dsize=4)

    # retrieve what type of descriptors it is.
    descriptors_type = image_describer.get('image_describer', {}).get('polymorphic_name', 'UNDEFINED')
    descriptors_props = {
        'SIFT_Image_describer': dict(type_name='SIFT', dtype=np.int32, dsize=128,
                                     keypoints_type=keypoints_type,
                                     metric_type='L2'),
        'AKAZE_Image_describer_SURF': dict(type_name='AKAZE', dtype=np.int32, dsize=128,
                                           keypoints_type=keypoints_type,
                                           metric_type='L2'),
    }.get(descriptors_type)
    if not descriptors_props:
        raise ValueError(f'conversion of {descriptors_type} descriptors not implemented.')
    kapture_descriptors = kapture.Descriptors(**descriptors_props)

    # populate regions files in openMVG directory
    # https://github.com/openMVG/openMVG/blob/master/src/openMVG/features/scalar_regions.hpp#L23
    image_names = kapture_data.records_camera.data_list()
    for image_name in image_names:
        openmvg_image_name = path.splitext(path.basename(image_name))[0]
        # keypoints
        keypoints_data = None
        openmvg_keypoints_filepath = path.join(openmvg_regions_directory_path, openmvg_image_name + '.feat')
        if path.isfile(openmvg_keypoints_filepath):
            # there is a keypoints file in openMVG, lets add it to kapture
            keypoints_data = np.loadtxt(openmvg_keypoints_filepath)
            assert keypoints_data.shape[1] == 4
            kapture_keypoints.add(image_name)
            # and convert file
            kapture_keypoints_filepath = get_keypoints_fullpath(keypoints_type,
                                                                kapture_path,
                                                                image_name)
            array_to_file(kapture_keypoints_filepath, keypoints_data)

        # descriptors
        openmvg_descriptors_filepath = path.join(openmvg_regions_directory_path, openmvg_image_name + '.desc')
        if path.isfile(openmvg_descriptors_filepath):
            assert path.isfile(openmvg_keypoints_filepath)
            # there is a keypoints file in openMVG, lets add it to kapture
            # assumes descriptors shape from keypoints_data shape
            descriptors_data_bytes = np.fromfile(openmvg_descriptors_filepath, dtype=np.uint8)
            nb_features = keypoints_data.shape[0] if keypoints_data is not None else 0
            descriptors_shape = descriptors_data_bytes[0:8].view(descriptors_props['dtype'])
            assert descriptors_shape[0] == nb_features
            descriptors_data = descriptors_data_bytes[8:].view(np.uint8).reshape((nb_features, 128))
            # descriptors_data.reshape((keypoints_data.shape[0], -1))
            kapture_descriptors.add(image_name)
            # and convert file
            kapture_descriptors_filepath = get_descriptors_fullpath(descriptors_type,
                                                                    kapture_path,
                                                                    image_name)
            array_to_file(kapture_descriptors_filepath, descriptors_data)

    kapture_data.keypoints = {keypoints_type: kapture_keypoints}
    kapture_data.descriptors = {descriptors_type: kapture_descriptors}


def _import_openmvg_matches(
        matches_file_path: str,
        kapture_data: kapture.Kapture,
        kapture_path: str):
    if kapture_data.records_camera is None:
        logger.warning('no images in records_camera, cannot import matches')
        return
    if kapture_data.keypoints is None:
        logger.warning('no keypoints, cannot import matches')
        return
    assert len(kapture_data.keypoints) == 1
    keypoints_type = next(iter(kapture_data.keypoints.keys()))

    # idx image1 idx image 2
    # nb pairs
    # pl1 pr1 pl2 pr2 ...
    openmvg_image_idx_to_kapture_image_name = {}
    matches = kapture.Matches()
    with open(matches_file_path, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            splits_idx = line.rstrip('\r\n').split()
            assert len(splits_idx) == 2
            idx_image1 = int(splits_idx[0])
            if idx_image1 in openmvg_image_idx_to_kapture_image_name:
                image_1 = openmvg_image_idx_to_kapture_image_name[idx_image1]
            else:
                if idx_image1 not in kapture_data.records_camera:
                    raise ValueError(f'{idx_image1} not in kapture_data.records_camera')
                assert len(kapture_data.records_camera[idx_image1]) == 1
                sensor_id = next(iter(kapture_data.records_camera[idx_image1].keys()))
                image_1 = kapture_data.records_camera.get(idx_image1)[sensor_id]
                openmvg_image_idx_to_kapture_image_name[idx_image1] = image_1

            idx_image2 = int(splits_idx[1])
            if idx_image2 in openmvg_image_idx_to_kapture_image_name:
                image_2 = openmvg_image_idx_to_kapture_image_name[idx_image2]
            else:
                if idx_image2 not in kapture_data.records_camera:
                    raise ValueError(f'{idx_image2} not in kapture_data.records_camera')
                assert len(kapture_data.records_camera[idx_image2]) == 1
                sensor_id = next(iter(kapture_data.records_camera[idx_image2].keys()))
                image_2 = kapture_data.records_camera.get(idx_image2)[sensor_id]
                openmvg_image_idx_to_kapture_image_name[idx_image2] = image_2

            swap_order = image_2 < image_1
            line = fid.readline()
            num_matches = int(line.rstrip('\r\n'))
            matches_array = np.empty((num_matches, 3), dtype=np.float)
            for i in range(num_matches):
                line = fid.readline()
                splits_kpts_idx = line.rstrip('\r\n').split()
                assert len(splits_kpts_idx) == 2
                if swap_order:
                    matches_array[i, 1] = int(splits_kpts_idx[0])
                    matches_array[i, 0] = int(splits_kpts_idx[1])
                else:
                    matches_array[i, 0] = int(splits_kpts_idx[0])
                    matches_array[i, 1] = int(splits_kpts_idx[1])
                matches_array[i, 2] = 1.0
            if swap_order:
                image_filename_pair = (image_2, image_1)
                matches.add(image_2, image_1)
            else:
                image_filename_pair = (image_1, image_2)
                matches.add(image_1, image_2)
            matches_filepath = get_matches_fullpath(image_filename_pair, keypoints_type, kapture_path)
            array_to_file(matches_filepath, matches_array)
    kapture_data.matches = {keypoints_type: matches}
