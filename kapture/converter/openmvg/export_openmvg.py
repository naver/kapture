# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Kapture to openmvg export functions.
"""

import json
import logging
import os
import os.path as path
import pathlib
import shutil
from typing import Dict, List, Union
import quaternion
# kapture
import kapture
import kapture.io.csv
from kapture.io.records import TransferAction, get_image_fullpath
import kapture.io.structure
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
# local
from .openmvg_commons import DEFAULT_JSON_FILE_NAME, SFM_DATA_VERSION, SFM_DATA_VERSION_NUMBER, ROOT_PATH, INTRINSICS,\
    VIEWS, VIEW_PRIORS, EXTRINSICS, KEY, VALUE, POLYMORPHIC_ID, PTR_WRAPPER, ID, DATA, LOCAL_PATH, FILENAME, ID_VIEW,\
    ID_INTRINSIC, ID_POSE, POLYMORPHIC_NAME, VALUE0, WIDTH, HEIGHT, FOCAL_LENGTH, PRINCIPAL_POINT,\
    DISTO_K1, DISTO_K3, DISTO_T2, FISHEYE, USE_POSE_CENTER_PRIOR, CENTER_WEIGHT, CENTER, USE_POSE_ROTATION_PRIOR,\
    ROTATION_WEIGHT, ROTATION, STRUCTURE, CONTROL_POINTS
from .openmvg_commons import CameraModel


logger = logging.getLogger('openmvg')  # Using global openmvg logger

NEW_ID_MASK = 1 << 31                  # 10000000 00000000 00000000 00000000
VIEW_SPECIAL_POLYMORPHIC_ID = 1 << 30  # 01000000 00000000 00000000 00000000
DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


def _get_data(camera_params: list) -> Dict:
    # w, h, f, cx, cy
    data: Dict[str, Union[int, float, List[float]]] = {WIDTH: int(camera_params[0]),
                                                       HEIGHT: int(camera_params[1]),
                                                       FOCAL_LENGTH: float(camera_params[2]),
                                                       PRINCIPAL_POINT: [float(camera_params[3]),
                                                                         float(camera_params[4])]}
    return data


def _get_intrinsic_pinhole(camera_params: list) -> Dict:
    # w, h, f, cx, cy
    return _get_data(camera_params)


def _get_intrinsic_pinhole_radial_k1(camera_params: list) -> Dict:
    data = _get_data(camera_params)
    # w, h, f, cx, cy, k
    data[DISTO_K1] = [float(camera_params[5])]
    return data


def _get_intrinsic_pinhole_radial_k3(camera_params: list) -> Dict:
    data = _get_data(camera_params)
    # w, h, f, cx, cy, k1, k2, k3
    data[DISTO_K3] = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7])]
    return data


def _get_intrinsic_pinhole_brown_t2(camera_params: list) -> Dict:
    # w, h, f, cx, cy, k1, k2, k3, t1, t2
    disto_t2 = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7]),
                float(camera_params[8]), float(camera_params[9])]
    return {VALUE0: _get_data(camera_params), DISTO_T2: disto_t2}


def _get_intrinsic_fisheye(camera_params: list) -> Dict:
    # w, h, f, cx, cy, k1, k2, k3, k4
    fisheye = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7]), float(camera_params[8])]
    return {VALUE0: _get_data(camera_params),
            FISHEYE: fisheye}


def load_kapture(kapture_path: str) -> kapture.Kapture:
    """
    Load the kapture data stored at the given path

    :param kapture_path: kapture data top directory
    :return: a kapture object
    """
    logger.info(f'Loading kapture data from {kapture_path}...')
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path)
    logger.info('loaded')
    assert isinstance(kapture_data, kapture.Kapture)
    return kapture_data


def kapture_to_openmvg(kapture_data: kapture.Kapture, kapture_path: str,
                       image_action: TransferAction, openmvg_path: str) -> Dict:
    """
    Convert the kapture data into an openMVG dataset stored as a dictionary.
    The format is defined here:
    https://openmvg.readthedocs.io/en/latest/software/SfM/SfM_OutputFormat/

    :param kapture_data: the kapture data
    :param kapture_path: top directory of the kapture data and the images
    :param image_action: action to apply on images: link, copy, move or do nothing.
    :param openmvg_path: top directory of the openmvg data and images
    :return: an SfM_data, the openmvg structure, stored as a dictionary ready to be serialized
    """

    assert kapture_data.cameras is not None
    assert kapture_data.records_camera is not None
    cameras = kapture_data.cameras
    # Check we don't have other sensors defined
    extra_sensor_number = len(kapture_data.sensors)-len(cameras)
    if extra_sensor_number > 0:
        logger.warning(f'We will ignore {extra_sensor_number} sensors that are not camera')
    records_camera = kapture_data.records_camera
    all_records_camera = list(kapture.flatten(records_camera))
    trajectories = kapture_data.trajectories

    # openmvg does not support rigs
    if kapture_data.rigs:
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

    # Compute root path and camera used in records
    sub_root_path: str = ''
    image_dirs = {}
    used_cameras = {}
    for _, cam_id, name in all_records_camera:
        used_cameras[cam_id] = cam_id
        img_dir = path.dirname(name)
        image_dirs[img_dir] = img_dir
    if len(image_dirs) > 1:
        # Find if they share a top path
        image_dirs_list = list(image_dirs.keys())
        sub_root_path = path.commonpath(image_dirs_list)
    elif len(image_dirs) == 1:
        sub_root_path = next(iter(image_dirs.keys()))
    if image_action == TransferAction.skip:
        root_path = kapture_path
    else:  # We will create a new hierarchy of images
        root_path = openmvg_path
    root_path = os.path.abspath(path.join(root_path, sub_root_path))
    if image_action == TransferAction.root_link:
        if not sub_root_path:
            # We can not link directly to the top destination openmvg directory
            # We need an additional level
            root_path = path.join(root_path, 'images')
        kapture_records_path = get_image_fullpath(kapture_path, sub_root_path)
        # Do a unique images directory link
        # openmvg_root_path -> kapture/<records_dir>/openmvg_top_images_directory
        # beware that the paths are reverted in the symlink call
        os.symlink(kapture_records_path, root_path)
    sfm_data = {SFM_DATA_VERSION: SFM_DATA_VERSION_NUMBER,
                ROOT_PATH: root_path}

    views = []
    intrinsics = []
    extrinsics = []
    polymorphic_id_current = 1
    ptr_wrapper_id_current = 1
    polymorphic_id_types = {}

    # process all cameras
    for cam_id, camera in cameras.items():
        # Ignore not used cameras
        if not used_cameras.get(cam_id):
            logger.warning(f'Skipping camera definition {cam_id} {camera.name} without recorded images.')
            continue
        cam_type = camera.camera_type
        camera_params = camera.camera_params
        if cam_type == kapture.CameraType.SIMPLE_PINHOLE:
            # w, h, f, cx, cy
            model_used = CameraModel.pinhole
            data = _get_intrinsic_pinhole(camera_params)
        elif cam_type == kapture.CameraType.PINHOLE:
            # w, h, f, cx, cy
            model_used = CameraModel.pinhole
            faked_params = [camera_params[0], camera_params[1],  # width height
                            (camera_params[2] + camera_params[3])/2,  # fx+fy/2 as f
                            camera_params[4], camera_params[5]]  # cx cy
            data = _get_intrinsic_pinhole(faked_params)
        elif cam_type == kapture.CameraType.SIMPLE_RADIAL:
            # w, h, f, cx, cy, k
            model_used = CameraModel.pinhole_radial_k1
            data = _get_intrinsic_pinhole_radial_k1(camera_params)
        elif cam_type == kapture.CameraType.RADIAL:
            # w, h, f, cx, cy, k1, k2, k3
            model_used = CameraModel.pinhole_radial_k3
            faked_params = [camera_params[0], camera_params[1],  # width height
                            camera_params[2],  # f
                            camera_params[3], camera_params[4],  # cx cy
                            camera_params[5], camera_params[6], 0  # k1, k2, k3
                            ]
            data = _get_intrinsic_pinhole_radial_k3(faked_params)
        elif cam_type == kapture.CameraType.FULL_OPENCV or cam_type == kapture.CameraType.OPENCV:
            # w, h, f, cx, cy, k1, k2, k3, t1, t2
            model_used = CameraModel.pinhole_brown_t2
            k3 = camera_params[10] if len(camera_params) > 10 else 0
            faked_params = [camera_params[0], camera_params[1],  # width height
                            (camera_params[2] + camera_params[3])/2,  # fx+fy/2 as f
                            camera_params[4], camera_params[5],  # cx cy
                            camera_params[6], camera_params[7], k3,  # k1, k2, k3
                            camera_params[8], camera_params[9]  # p1, p2 (=t1, t2)
                            ]
            data = _get_intrinsic_pinhole_brown_t2(faked_params)
        elif cam_type == kapture.CameraType.OPENCV_FISHEYE:
            logger.warning('OpenCV fisheye model is not compatible with OpenMVG. Forcing distortion to 0')
            # w, h, f, cx, cy, k1, k2, k3, k4
            model_used = CameraModel.fisheye
            faked_params = [camera_params[0], camera_params[1],  # width height
                            (camera_params[2] + camera_params[3]) / 2,  # fx+fy/2 as f
                            camera_params[4], camera_params[5],  # cx cy
                            0, 0,  # k1, k2
                            0, 0  # k3, k4
                            ]
            data = _get_intrinsic_fisheye(faked_params)
        elif cam_type == kapture.CameraType.RADIAL_FISHEYE or cam_type == kapture.CameraType.SIMPLE_RADIAL_FISHEYE:
            logger.warning('OpenCV fisheye model is not compatible with OpenMVG. Forcing distortion to 0')
            # w, h, f, cx, cy, k1, k2, k3, k4
            model_used = CameraModel.fisheye
            faked_params = [camera_params[0], camera_params[1],  # width height
                            camera_params[2],  # f
                            camera_params[3], camera_params[4],  # cx cy
                            0, 0,  # k1, k2
                            0, 0  # k3, k4
                            ]
            data = _get_intrinsic_fisheye(faked_params)
        elif cam_type == kapture.CameraType.UNKNOWN_CAMERA:
            logger.info(f'Camera {cam_id}: Unknown camera model, using simple radial')
            # Choose simple radial model, to allow openMVG to determine distortion param
            # w, h, f, cx, cy, k
            model_used = CameraModel.pinhole_radial_k1
            faked_params = [camera_params[0], camera_params[1],  # width height
                            max(camera_params[0], camera_params[1])*DEFAULT_FOCAL_LENGTH_FACTOR,  # max(w,h)*1.2 as f
                            int(camera_params[0]/2), int(camera_params[1]/2),  # cx cy
                            0.0]  # k1
            data = _get_intrinsic_pinhole_radial_k1(faked_params)
        else:
            raise ValueError(f'Camera model {cam_type.value} not supported')

        intrinsic = {}
        if model_used not in polymorphic_id_types:
            # if this is the first time model_used is encountered
            # set the first bit of polymorphic_id_current to 1
            intrinsic[POLYMORPHIC_ID] = polymorphic_id_current | NEW_ID_MASK
            intrinsic[POLYMORPHIC_NAME] = model_used.name
            polymorphic_id_types[model_used] = polymorphic_id_current
            polymorphic_id_current += 1
        else:
            intrinsic[POLYMORPHIC_ID] = polymorphic_id_types[model_used]

        # it is assumed that this camera is only encountered once
        # set the first bit of ptr_wrapper_id_current to 1
        data_wrapper = {ID: ptr_wrapper_id_current | NEW_ID_MASK,
                        DATA: data}
        ptr_wrapper_id_current += 1

        intrinsic[PTR_WRAPPER] = data_wrapper
        intrinsics.append({KEY: cam_id, VALUE: intrinsic})

    global_timestamp = 0

    # process all images
    for timestamp, cam_id, kapture_name in all_records_camera:
        local_path = path.dirname(path.relpath(kapture_name, sub_root_path))
        filename = path.basename(kapture_name)
        if image_action != TransferAction.skip and image_action != TransferAction.root_link:
            # Process the image action
            src_path = get_image_fullpath(kapture_path, kapture_name)
            dst_path = path.join(root_path, local_path, filename)
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
                # Link
                if image_action == TransferAction.link_relative:
                    # Compute relative path
                    src_path = path.relpath(src_path, dst_dir)
                os.symlink(src_path, dst_path)

        camera_params = cameras[cam_id].camera_params
        view_data = {LOCAL_PATH: local_path,
                     FILENAME: filename,
                     WIDTH: int(camera_params[0]),
                     HEIGHT: int(camera_params[1]),
                     ID_VIEW: global_timestamp,
                     ID_INTRINSIC: cam_id,
                     ID_POSE: global_timestamp}

        view = {}
        # retrieve image pose from trajectories
        if timestamp not in trajectories:
            view[POLYMORPHIC_ID] = VIEW_SPECIAL_POLYMORPHIC_ID
        else:
            # there is a pose for that timestamp
            # The poses are stored both as priors (in the 'views' table) and as known poses (in the 'extrinsics' table)
            assert cam_id in trajectories[timestamp]
            if VIEW_PRIORS not in polymorphic_id_types:
                # if this is the first time view_priors is encountered
                # set the first bit of polymorphic_id_current to 1
                view[POLYMORPHIC_ID] = polymorphic_id_current | NEW_ID_MASK
                view[POLYMORPHIC_NAME] = VIEW_PRIORS
                polymorphic_id_types[VIEW_PRIORS] = polymorphic_id_current
                polymorphic_id_current += 1
            else:
                view[POLYMORPHIC_ID] = polymorphic_id_types[VIEW_PRIORS]

            pose_tr = trajectories[timestamp].get(cam_id)
            prior_q = pose_tr.r
            prior_t = pose_tr.inverse().t_raw
            pose_data = {CENTER: prior_t,
                         ROTATION: quaternion.as_rotation_matrix(prior_q).tolist()}

            view_data[USE_POSE_CENTER_PRIOR] = True
            view_data[CENTER_WEIGHT] = [1.0, 1.0, 1.0]
            view_data[CENTER] = prior_t
            view_data[USE_POSE_ROTATION_PRIOR] = True
            view_data[ROTATION_WEIGHT] = 1.0
            view_data[ROTATION] = pose_data[ROTATION]
            extrinsics.append({KEY: global_timestamp, VALUE: pose_data})

        # it is assumed that this view is only encountered once
        # set the first bit of ptr_wrapper_id_current to 1
        view_wrapper = {ID: ptr_wrapper_id_current | NEW_ID_MASK,
                        DATA: view_data}
        ptr_wrapper_id_current += 1

        view[PTR_WRAPPER] = view_wrapper
        views.append({KEY: global_timestamp, VALUE: view})

        global_timestamp += 1

    sfm_data[VIEWS] = views
    sfm_data[INTRINSICS] = intrinsics
    sfm_data[EXTRINSICS] = extrinsics
    sfm_data[STRUCTURE] = []
    sfm_data[CONTROL_POINTS] = []

    return sfm_data


def export_openmvg(kapture_path: str, openmvg_path: str,
                   image_action: TransferAction, force: bool = False) -> None:
    """
    Export the kapture data to an openMVG JSON file.
    If the openmvg_path is a directory, it will create a JSON file (using the default name sfm_data.json)
    in that directory.

    :param kapture_path: full path to the top kapture directory
    :param openmvg_path: path of the file or directory where to store the data as JSON
    :param image_action: an action to apply on the images: relative linking, absolute linking, copy or move. Or top
     directory linking or skip to do nothing.
    :param force: if true, will remove existing openMVG data without prompting the user.
    """

    if path.isdir(openmvg_path):  # Existing directory
        json_file = path.join(openmvg_path, DEFAULT_JSON_FILE_NAME)
    else:
        file_ext = path.splitext(openmvg_path)[1]
        if len(file_ext) == 0:  # No extension: -> new directory
            json_file = path.join(openmvg_path, DEFAULT_JSON_FILE_NAME)
        elif file_ext.lower() != '.json':
            logger.warning(f'Creating output directory with file extension {file_ext}')
            json_file = path.join(openmvg_path, DEFAULT_JSON_FILE_NAME)
        else:  # Json file
            json_file = openmvg_path
    json_dir = path.dirname(json_file)
    safe_remove_file(json_file, force)
    if path.exists(json_file):
        raise ValueError(f'{json_file} file already exist')
    if image_action != TransferAction.skip and path.exists(json_dir) and any(pathlib.Path(json_dir).iterdir()):
        safe_remove_any_path(json_dir, force)
        if path.isdir(json_dir):
            raise ValueError(f'Images directory {json_dir} exist with remaining files')
    os.makedirs(json_dir, exist_ok=True)

    kapture_data = load_kapture(kapture_path)
    openmvg_data = kapture_to_openmvg(kapture_data, kapture_path, image_action, json_dir)
    logger.info(f'Saving to openmvg file {json_file}')
    with open(json_file, "w") as fid:
        json.dump(openmvg_data, fid, indent=4)
