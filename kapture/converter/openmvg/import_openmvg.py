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
from typing import Dict, Union
# kapture
import kapture
import kapture.io.csv as kcsv
import kapture.io.structure
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.utils.paths import path_secure
# local
from .openmvg_commons import DEFAULT_JSON_FILE_NAME, ROOT_PATH, INTRINSICS, VIEWS, EXTRINSICS,\
    KEY, VALUE, POLYMORPHIC_ID, PTR_WRAPPER, DATA, LOCAL_PATH, FILENAME, ID_VIEW, ID_INTRINSIC,\
    ID_POSE, POLYMORPHIC_NAME, VALUE0, WIDTH, HEIGHT, FOCAL_LENGTH, PRINCIPAL_POINT, DISTO_K1, DISTO_K3, DISTO_T2, \
    ROTATION, CENTER
from .openmvg_commons import CameraModel

logger = logging.getLogger('openmvg')  # Using global openmvg logger


def import_openmvg(input_path: str,
                   kapture_path: str,
                   image_action: TransferAction,
                   force_overwrite_existing: bool = False) -> None:
    """
    Converts an openMVG JSON file to a kapture directory.
    If an image action is provided (link, copy or move), links to the image files are created,
    or the image files are copied or moved.

    :param input_path: path to the openMVG file (or its directory)
    :param kapture_path: path to the kapture directory where the data will be exported
    :param image_action: action to apply to the images
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    """

    os.makedirs(kapture_path, exist_ok=True)
    kapture.io.structure.delete_existing_kapture_files(kapture_path, force_overwrite_existing)
    if path.isdir(input_path):
        input_file = path.join(input_path, DEFAULT_JSON_FILE_NAME)
    else:
        input_file = input_path
    if not path.isfile(input_file):
        raise ValueError(f'OpenMVG JSON file {input_file} does not exist')

    logger.info(f'Loading openmvg file {input_file}')

    with open(input_file, 'r') as f:
        input_json = json.load(f)
        kapture_data = openmvg_to_kapture(input_json, kapture_path, image_action)
        logger.info(f'Saving to kapture {kapture_path}')
        kcsv.kapture_to_dir(kapture_path, kapture_data)


GET_ID_MASK = 2147483647  # 01111111 11111111 11111111 11111111
ID_POSE_NOT_LOCALIZED = 4294967295  # 11111111 11111111 11111111 11111111


def openmvg_to_kapture(input_json: Dict[str, Union[str, Dict]],
                       kapture_images_path: str,
                       image_action=TransferAction.skip) -> kapture.Kapture:
    """
    Convert an openMVG structure to a kapture object. Also copy, move or link the images files if necessary.

    :param input_json: the openmvg JSON parsed as a dictionary
    :param kapture_images_path: top directory to create the kapture images tree
    :param image_action: action to apply on images: link, copy, move or do nothing.
    :return: the constructed kapture object
    """

    root_path: str = ''

    if input_json[ROOT_PATH]:
        root_path = input_json[ROOT_PATH]
    elif image_action == TransferAction.skip:
        logger.warning("No root_path in input file")
    else:  # It is needed to execute an action with the image file
        raise ValueError(f"Missing root_path to do image action '{image_action.name}'")
    openmvg_images_dir = path.basename(root_path)

    # Imports all the data from the json file to kapture objects
    kapture_cameras = _import_cameras(input_json)
    device_identifiers = {int: str}  # Pose id -> device id
    timestamp_for_pose = {int: int}  # Pose id -> timestamp
    # Imports the images as records_camera, but also fill in the devices_identifiers and timestamp_for_pose dictionaries
    records_camera = _import_images(input_json, image_action, kapture_images_path, openmvg_images_dir, root_path,
                                    device_identifiers, timestamp_for_pose)
    trajectories = _import_trajectories(input_json, device_identifiers, timestamp_for_pose)

    kapture_data = kapture.Kapture(sensors=kapture_cameras, records_camera=records_camera, trajectories=trajectories)
    return kapture_data


def _import_cameras(input_json) -> kapture.Sensors:  # noqa: C901
    kapture_cameras = kapture.Sensors()
    if input_json.get(INTRINSICS):
        polymorphic_id_to_value = {}
        logger.info('Importing intrinsics')
        for sensor in input_json[INTRINSICS]:
            value = sensor[VALUE]
            if POLYMORPHIC_NAME in value:
                # new type name: store it for next instances
                polymorphic_id = value[POLYMORPHIC_ID] & GET_ID_MASK
                polymorphic_id_to_value[polymorphic_id] = value[POLYMORPHIC_NAME]
                logger.debug("New camera_type: " + polymorphic_id_to_value[polymorphic_id])
            else:
                if POLYMORPHIC_ID not in value:
                    raise ValueError(f'{POLYMORPHIC_ID} is missing (intrinsics)')
                polymorphic_id = value[POLYMORPHIC_ID]

            if polymorphic_id not in polymorphic_id_to_value:
                raise ValueError(f'Unknown polymorphic_id {polymorphic_id}')

            camera_model = CameraModel(polymorphic_id_to_value[polymorphic_id])
            camera_data = value[PTR_WRAPPER][DATA]

            if camera_model == CameraModel.pinhole:
                # w, h, f, cx, cy
                camera = kapture.Camera(kapture.CameraType.SIMPLE_PINHOLE, [
                    int(camera_data[WIDTH]),
                    int(camera_data[HEIGHT]),
                    camera_data[FOCAL_LENGTH],
                    camera_data[PRINCIPAL_POINT][0],
                    camera_data[PRINCIPAL_POINT][1],
                ])
            elif camera_model == CameraModel.pinhole_radial_k1:
                # w, h, f, cx, cy, k
                camera = kapture.Camera(kapture.CameraType.SIMPLE_RADIAL, [
                    int(camera_data[WIDTH]),
                    int(camera_data[HEIGHT]),
                    camera_data[FOCAL_LENGTH],
                    camera_data[PRINCIPAL_POINT][0],
                    camera_data[PRINCIPAL_POINT][1],
                    camera_data[DISTO_K1][0]
                ])
            elif camera_model == CameraModel.pinhole_radial_k3:
                # w, h, f, cx, cy, k1, k2, k3
                camera = kapture.Camera(kapture.CameraType.RADIAL, [
                    int(camera_data[WIDTH]),
                    int(camera_data[HEIGHT]),
                    camera_data[FOCAL_LENGTH],
                    camera_data[PRINCIPAL_POINT][0],
                    camera_data[PRINCIPAL_POINT][1],
                    camera_data[DISTO_K3][0],
                    camera_data[DISTO_K3][1]
                ])
                # camera_data["disto_k3"][2] ignored: radial model has two distortion param, while openMVG's has three
            elif camera_model == CameraModel.pinhole_brown_t2:
                # w, h, f, cx, cy, k1, k2, k3, t1, t2
                if float(camera_data[DISTO_T2][2]) != 0:
                    # if k3 not null, use FULL_OPENCV, otherwise OPENCV
                    # w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                    value0 = camera_data[VALUE0]
                    disto_t2 = camera_data[DISTO_T2]
                    camera = kapture.Camera(kapture.CameraType.FULL_OPENCV, [
                        int(value0[WIDTH]),
                        int(value0[HEIGHT]),
                        value0[FOCAL_LENGTH],
                        value0[FOCAL_LENGTH],
                        value0[PRINCIPAL_POINT][0],
                        value0[PRINCIPAL_POINT][1],
                        disto_t2[0], disto_t2[1], disto_t2[3], disto_t2[4], disto_t2[2],
                        0, 0, 0
                    ])
                else:
                    # w, h, fx, fy, cx, cy, k1, k2, p1, p2
                    value0 = camera_data[VALUE0]
                    disto_t2 = camera_data[DISTO_T2]
                    camera = kapture.Camera(kapture.CameraType.OPENCV, [
                        int(value0[WIDTH]),
                        int(value0[HEIGHT]),
                        value0[FOCAL_LENGTH],
                        value0[FOCAL_LENGTH],
                        value0[PRINCIPAL_POINT][0],
                        value0[PRINCIPAL_POINT][1],
                        disto_t2[0], disto_t2[1], disto_t2[3], disto_t2[4]])
            elif camera_model == CameraModel.fisheye:
                logger.warning(
                    "OpenMVG fisheye models are not compatible with OpenCV."
                    " Using SIMPLE_RADIAL_FISHEYE and forcing distortion to 0")
                # w, h, f, cx, cy, k
                value0 = camera_data[VALUE0]
                camera = kapture.Camera(kapture.CameraType.SIMPLE_RADIAL_FISHEYE, [
                    int(value0[WIDTH]),
                    int(value0[HEIGHT]),
                    value0[FOCAL_LENGTH],
                    value0[PRINCIPAL_POINT][0],
                    value0[PRINCIPAL_POINT][1],
                    0])
            else:
                raise ValueError(f'Camera model {camera_model} not supported')

            kapture_cameras[str(sensor[KEY])] = camera

    return kapture_cameras


def _import_images(input_json, image_action, kapture_images_path, openmvg_images_dir, root_path,
                   device_identifiers, timestamp_for_pose):
    records_camera = kapture.RecordsCamera()
    if input_json.get(VIEWS):
        views = input_json[VIEWS]
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
            input_data = view[VALUE][PTR_WRAPPER][DATA]
            pose_id = input_data[ID_POSE]
            # All two values should be the same (?)
            if input_data[ID_VIEW]:
                timestamp = input_data[ID_VIEW]
            else:
                timestamp = view[KEY]
            device_id = str(input_data[ID_INTRINSIC])  # device_id must be a string for kapture
            device_identifiers[pose_id] = device_id
            timestamp_for_pose[pose_id] = timestamp

            kapture_filename = _import_image_file(input_data, openmvg_images_dir, root_path,
                                                  kapture_images_path, image_action)

            progress_bar and progress_bar.update(1)

            key = (timestamp, device_id)  # tuple of int,str
            records_camera[key] = path_secure(kapture_filename)
        progress_bar and progress_bar.close()
    return records_camera


def _import_image_file(input_data, openmvg_images_dir, root_path, kapture_images_path, image_action) -> str:
    # Add the common openmvg images directory in front of the filename
    filename: str
    if input_data.get(LOCAL_PATH):
        filename = path.join(input_data[LOCAL_PATH], input_data[FILENAME])
    else:
        filename = input_data[FILENAME]
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


def _import_trajectories(input_json, device_identifiers, timestamp_for_pose):
    trajectories = kapture.Trajectories()
    if input_json.get(EXTRINSICS):
        extrinsics = input_json[EXTRINSICS]
        logger.info(f'Importing {len(extrinsics)} extrinsics -> trajectories')
        for pose in extrinsics:
            pose_id = pose[KEY]
            center = pose[VALUE][CENTER]
            rotation = pose[VALUE][ROTATION]
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
