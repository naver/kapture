# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Kapture to openmvg export functions.
"""

import json
import logging
import os
import os.path as path
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import quaternion

# kapture
import kapture
from kapture.core.Trajectories import rigs_remove_inplace
import kapture.io.csv
from kapture.io.binary import TransferAction, transfer_files_from_dir, array_to_file
from kapture.io.features import keypoints_to_filepaths, image_keypoints_from_file, get_keypoints_fullpath
from kapture.io.features import descriptors_to_filepaths, image_descriptors_from_file
from kapture.io.features import matches_to_filepaths, image_matches_from_file
from kapture.io.records import get_image_fullpath
import kapture.io.structure
from kapture.io.tar import TarCollection
from kapture.utils.Collections import try_get_only_key_from_collection
from kapture.utils.paths import safe_remove_file
# local
from .openmvg_commons import JSON_KEY, OPENMVG_SFM_DATA_VERSION_NUMBER, OPENMVG_DEFAULT_REGIONS_FILE_NAME
from .openmvg_commons import OPENMVG_DESC_HEADER_DTYPE, OPENMVG_DESC_HEADER_BYTES_NUMBER
from .openmvg_commons import CameraModel

logger = logging.getLogger('openmvg')  # Using global openmvg logger

NEW_ID_MASK = 1 << 31  # 10000000 00000000 00000000 00000000
DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


class CerealPointerRegistry:
    """
    A Cereal registry ticketing system.
    """
    NULL_ID = 1 << 30  # 01000000 00000000 00000000 00000000

    def __init__(self, value_key, id_key):
        self._value_key = value_key
        self._id_key = id_key
        self._ids = {}
        self._id_current = 1

    def get_ids_dict(self, value):
        """ :return either id if known or id + name if new """
        if isinstance(value, dict) or value not in self._ids:
            # if this is the first time or a dict
            new_id = self._id_current
            if not isinstance(value, dict):
                self._ids[value] = new_id
            self._id_current += 1
            return {
                self._id_key: new_id | NEW_ID_MASK,
                self._value_key: value,
            }
        else:
            return {
                self._id_key: self._ids[value]
            }


def _get_camera_common_data(camera_params: list) -> Dict:
    # w, h, f, cx, cy
    data: Dict[str, Union[int, float, List[float]]] = {JSON_KEY.WIDTH: int(camera_params[0]),
                                                       JSON_KEY.HEIGHT: int(camera_params[1]),
                                                       JSON_KEY.FOCAL_LENGTH: float(camera_params[2]),
                                                       JSON_KEY.PRINCIPAL_POINT: [float(camera_params[3]),
                                                                                  float(camera_params[4])]}
    return data


def _get_intrinsic_pinhole(camera_params: list) -> Dict:
    # w, h, f, cx, cy
    return _get_camera_common_data(camera_params)


def _get_intrinsic_pinhole_radial_k1(camera_params: list) -> Dict:
    data = _get_camera_common_data(camera_params)
    # w, h, f, cx, cy, k
    data[JSON_KEY.DISTO_K1] = [float(camera_params[5])]
    return data


def _get_intrinsic_pinhole_radial_k3(camera_params: list) -> Dict:
    data = _get_camera_common_data(camera_params)
    # w, h, f, cx, cy, k1, k2, k3
    data[JSON_KEY.DISTO_K3] = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7])]
    return data


def _get_intrinsic_pinhole_brown_t2(camera_params: list) -> Dict:
    # w, h, f, cx, cy, k1, k2, k3, t1, t2
    disto_t2 = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7]),
                float(camera_params[8]), float(camera_params[9])]
    return {JSON_KEY.VALUE0: _get_camera_common_data(camera_params), JSON_KEY.DISTO_T2: disto_t2}


def _get_intrinsic_fisheye(camera_params: list) -> Dict:
    # w, h, f, cx, cy, k1, k2, k3, k4
    fisheye = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7]), float(camera_params[8])]
    return {JSON_KEY.VALUE0: _get_camera_common_data(camera_params),
            JSON_KEY.FISHEYE: fisheye}


def _compute_openmvg_id(kapture_id: str, kapture_to_openmvg_ids: Dict[str, int]) -> None:
    """ Compute a valid openmvg id for the given kapture id.
    It keeps kapture_to_openmvg_ids up to date, and ensure there is no collision.

    :return: openmvg id
    """

    if kapture_id not in kapture_to_openmvg_ids:
        # its not defined, then, make up one
        last_known_openmvg_id = max(list(kapture_to_openmvg_ids.values())) if len(kapture_to_openmvg_ids) > 0 else -1
        assert type(last_known_openmvg_id) == int
        openmvg_id = last_known_openmvg_id + 1
        assert openmvg_id not in kapture_to_openmvg_ids.values()
        kapture_to_openmvg_ids[kapture_id] = openmvg_id


def _get_openmvg_image_path(kapture_image_name: str, flatten_path: bool = False):
    """ the openmvg image sub path corresponding to the given kapture one. """
    return kapture_image_name if not flatten_path else kapture_image_name.replace('/', '_')


def _export_openmvg_intrinsics(
        kapture_cameras: Dict[str, kapture.Camera],
        kapture_to_openmvg_cam_ids: Dict[str, int],
        polymorphic_registry: CerealPointerRegistry,
        ptr_wrapper_registry: CerealPointerRegistry,
) -> List:
    """
    Exports the given kapture cameras to the openMVG sfm_data structure.
    In openMVG, cameras are referred as Intrinsics camera internal parameters.

    :param kapture_cameras: input kapture cameras to be exported (only if used).
    :param kapture_to_openmvg_cam_ids: dict that maps kapture camera ids to openMVG camera ids.
    :param polymorphic_registry: polymorphic IDs status
    :param ptr_wrapper_registry: polymorphic IDs status
    :return: intrinsics to be serialized
    """
    openmvg_intrinsics = []
    # process all cameras
    for kapture_cam_id, kapture_camera in kapture_cameras.items():
        openmvg_camera_id = kapture_to_openmvg_cam_ids.get(kapture_cam_id)
        if openmvg_camera_id is None:
            # this cameras is not used, skip it to make openMVG happy
            logger.debug(f'skip intrinsic parameters for camera {kapture_cam_id}')
            continue

        kapture_cam_type = kapture_camera.camera_type
        kapture_camera_params = kapture_camera.camera_params
        if kapture_cam_type == kapture.CameraType.SIMPLE_PINHOLE:
            # w, h, f, cx, cy
            openmvg_cam_type = CameraModel.pinhole
            data = _get_intrinsic_pinhole(kapture_camera_params)
        elif kapture_cam_type == kapture.CameraType.PINHOLE:
            # w, h, f, cx, cy
            openmvg_cam_type = CameraModel.pinhole
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            (kapture_camera_params[2] + kapture_camera_params[3]) / 2,  # fx+fy/2 as f
                            kapture_camera_params[4], kapture_camera_params[5]]  # cx cy
            data = _get_intrinsic_pinhole(faked_params)
        elif kapture_cam_type == kapture.CameraType.SIMPLE_RADIAL:
            # w, h, f, cx, cy, k
            openmvg_cam_type = CameraModel.pinhole_radial_k1
            data = _get_intrinsic_pinhole_radial_k1(kapture_camera_params)
        elif kapture_cam_type == kapture.CameraType.RADIAL:
            # w, h, f, cx, cy, k1, k2, k3
            openmvg_cam_type = CameraModel.pinhole_radial_k3
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            kapture_camera_params[2],  # f
                            kapture_camera_params[3], kapture_camera_params[4],  # cx cy
                            kapture_camera_params[5], kapture_camera_params[6], 0  # k1, k2, k3
                            ]
            data = _get_intrinsic_pinhole_radial_k3(faked_params)
        elif kapture_cam_type == kapture.CameraType.FULL_OPENCV or kapture_cam_type == kapture.CameraType.OPENCV:
            # w, h, f, cx, cy, k1, k2, k3, t1, t2
            openmvg_cam_type = CameraModel.pinhole_brown_t2
            k3 = kapture_camera_params[10] if len(kapture_camera_params) > 10 else 0
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            (kapture_camera_params[2] + kapture_camera_params[3]) / 2,  # fx+fy/2 as f
                            kapture_camera_params[4], kapture_camera_params[5],  # cx cy
                            kapture_camera_params[6], kapture_camera_params[7], k3,  # k1, k2, k3
                            kapture_camera_params[8], kapture_camera_params[9]  # p1, p2 (=t1, t2)
                            ]
            data = _get_intrinsic_pinhole_brown_t2(faked_params)
        elif kapture_cam_type == kapture.CameraType.OPENCV_FISHEYE:
            logger.warning('OpenCV fisheye model is not compatible with OpenMVG. Forcing distortion to 0')
            # w, h, f, cx, cy, k1, k2, k3, k4
            openmvg_cam_type = CameraModel.fisheye
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            (kapture_camera_params[2] + kapture_camera_params[3]) / 2,  # fx+fy/2 as f
                            kapture_camera_params[4], kapture_camera_params[5],  # cx cy
                            0, 0,  # k1, k2
                            0, 0  # k3, k4
                            ]
            data = _get_intrinsic_fisheye(faked_params)
        elif kapture_cam_type == kapture.CameraType.RADIAL_FISHEYE or \
                kapture_cam_type == kapture.CameraType.SIMPLE_RADIAL_FISHEYE:
            logger.warning('OpenCV fisheye model is not compatible with OpenMVG. Forcing distortion to 0')
            # w, h, f, cx, cy, k1, k2, k3, k4
            openmvg_cam_type = CameraModel.fisheye
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            kapture_camera_params[2],  # f
                            kapture_camera_params[3], kapture_camera_params[4],  # cx cy
                            0, 0,  # k1, k2
                            0, 0  # k3, k4
                            ]
            data = _get_intrinsic_fisheye(faked_params)
        elif kapture_cam_type == kapture.CameraType.UNKNOWN_CAMERA:
            logger.info(f'Camera {kapture_cam_id}: Unknown camera model, using simple radial')
            # Choose simple radial model, to allow openMVG to determine distortion param
            # w, h, f, cx, cy, k
            openmvg_cam_type = CameraModel.pinhole_radial_k1
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            max(kapture_camera_params[0], kapture_camera_params[1]) * DEFAULT_FOCAL_LENGTH_FACTOR,
                            # max(w,h)*1.2 as f
                            int(kapture_camera_params[0] / 2), int(kapture_camera_params[1] / 2),  # cx cy
                            0.0]  # k1
            data = _get_intrinsic_pinhole_radial_k1(faked_params)
        else:
            raise ValueError(f'Camera model {kapture_cam_type.value} not supported')

        intrinsic = {}
        cam_type_poly_id = polymorphic_registry.get_ids_dict(openmvg_cam_type.name)
        intrinsic.update(cam_type_poly_id)

        # it is assumed that this camera is only encountered once
        # set the first bit of ptr_wrapper_id_current to 1
        data_wrapper = ptr_wrapper_registry.get_ids_dict(data)

        intrinsic[JSON_KEY.PTR_WRAPPER] = data_wrapper
        openmvg_intrinsics.append({JSON_KEY.KEY: openmvg_camera_id, JSON_KEY.VALUE: intrinsic})

    return openmvg_intrinsics


def _export_openmvg_views(
        kapture_cameras: Dict[str, kapture.Camera],
        kapture_images_data: List[Tuple[int, str, str]],
        kapture_trajectories: kapture.Trajectories,
        kapture_to_openmvg_cam_ids: Dict[str, int],
        kapture_to_openmvg_view_ids: Dict[str, int],
        polymorphic_registry: CerealPointerRegistry,
        ptr_wrapper_registry: CerealPointerRegistry,
        sub_root_path: str,
        image_path_flatten: bool,
) -> List:
    """

    :param kapture_cameras:
    :param kapture_images_data: list of all (timestamp, camera_id, image_name) tuples
    :param kapture_trajectories:
    :param kapture_to_openmvg_cam_ids: dict that maps kapture camera ids to openMVG camera ids.
    :param kapture_to_openmvg_view_ids: dict that maps kapture image names to openMVG view ids.
    :param polymorphic_registry: input/output polymorphic IDs status
    :param ptr_wrapper_registry: input/output polymorphic IDs status
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :return: views to be serialized
    """
    views = []

    """
    fills an array of views like :
    {
            "key": 0,
            "value": {
                "polymorphic_id": 2147483650, "polymorphic_name": "view_priors", "ptr_wrapper": {
                    "id": 2147488881,
                    "data": {
                        "local_path": "db",
                        "filename": "1.jpg",
                        "width": 1600,
                        "height": 1063,
                        "id_view": 0,
                        "id_intrinsic": 3156,
                        "id_pose": 0,
                and optionally
                        "use_pose_center_prior": true, "center_weight": [1.0,1.0,1.0], "center": [...],
                        "use_pose_rotation_prior": true, "rotation_weight": 1.0, "rotation": [[...],[...],[...]]
                    }
                }
            }
        },
    """
    # process all images
    for timestamp, kapture_cam_id, kapture_image_name in kapture_images_data:
        openmvg_cam_id = kapture_to_openmvg_cam_ids[kapture_cam_id]
        openmvg_view_id = kapture_to_openmvg_view_ids[kapture_image_name]
        if sub_root_path:
            openmvg_image_filepath = _get_openmvg_image_path(path.relpath(kapture_image_name, sub_root_path),
                                                             image_path_flatten)
        else:
            openmvg_image_filepath = _get_openmvg_image_path(kapture_image_name, image_path_flatten)
        openmvg_image_filename = path.basename(openmvg_image_filepath)
        openmvg_image_local_path = path.dirname(openmvg_image_filepath)
        kapture_camera_params = kapture_cameras[kapture_cam_id].camera_params
        view_data = {JSON_KEY.LOCAL_PATH: openmvg_image_local_path,
                     JSON_KEY.FILENAME: openmvg_image_filename,
                     JSON_KEY.WIDTH: int(kapture_camera_params[0]),
                     JSON_KEY.HEIGHT: int(kapture_camera_params[1]),
                     JSON_KEY.ID_VIEW: openmvg_view_id,
                     JSON_KEY.ID_INTRINSIC: openmvg_cam_id,
                     JSON_KEY.ID_POSE: openmvg_view_id}

        view = {}
        # retrieve image pose from trajectories
        if timestamp not in kapture_trajectories:
            view[JSON_KEY.POLYMORPHIC_ID] = CerealPointerRegistry.NULL_ID
        else:
            # there is a pose for that timestamp
            # The poses are stored both as priors (in the 'views' table) and as known poses (in the 'extrinsics' table)
            assert kapture_cam_id in kapture_trajectories[timestamp]
            view_priors_id = polymorphic_registry.get_ids_dict(JSON_KEY.VIEW_PRIORS)
            view.update(view_priors_id)

            pose_tr = kapture_trajectories[timestamp].get(kapture_cam_id)
            prior_q = pose_tr.r
            prior_t = pose_tr.inverse().t_raw
            pose_data = {JSON_KEY.CENTER: prior_t,
                         JSON_KEY.ROTATION: quaternion.as_rotation_matrix(prior_q).tolist()}

            view_data[JSON_KEY.USE_POSE_CENTER_PRIOR] = True
            view_data[JSON_KEY.CENTER_WEIGHT] = [1.0, 1.0, 1.0]
            view_data[JSON_KEY.CENTER] = prior_t
            view_data[JSON_KEY.USE_POSE_ROTATION_PRIOR] = True
            view_data[JSON_KEY.ROTATION_WEIGHT] = 1.0
            view_data[JSON_KEY.ROTATION] = pose_data[JSON_KEY.ROTATION]

        # it is assumed that this view is only encountered once
        view_wrapper = ptr_wrapper_registry.get_ids_dict(view_data)
        view[JSON_KEY.PTR_WRAPPER] = view_wrapper
        views.append({JSON_KEY.KEY: openmvg_view_id, JSON_KEY.VALUE: view})

    return views


def _export_openmvg_extrinsics(
        kapture_images_data: List[Tuple[int, str, str]],
        kapture_trajectories: kapture.Trajectories,
        kapture_to_openmvg_view_ids: Dict[str, int],
) -> List:
    """

    :param kapture_images_data: all kapture images
    :param kapture_trajectories: all kapture poses
    :param kapture_to_openmvg_view_ids: dict that maps kapture image ids to openMVG view ids.
    :return: extrinsics to be serialized
    """
    extrinsics = []
    # process all images
    for timestamp, kapture_cam_id, kapture_image_name in kapture_images_data:
        openmvg_view_id = kapture_to_openmvg_view_ids.get(kapture_image_name)
        if openmvg_view_id is None:
            # this pose corresponds to no views (orphan), openMVG does not want it.
            logger.debug(f'skipping extrinsic for {kapture_image_name} (not referenced in views).')
            continue

        # retrieve image pose from trajectories
        if timestamp in kapture_trajectories:
            # there is a pose for that timestamp
            # The poses are stored both as priors (in the 'views' table) and as known poses (in the 'extrinsics' table)
            assert kapture_cam_id in kapture_trajectories[timestamp]
            pose_tr = kapture_trajectories[timestamp].get(kapture_cam_id)
            prior_q = pose_tr.r
            prior_t = pose_tr.inverse().t_raw
            pose_data = {JSON_KEY.CENTER: prior_t,
                         JSON_KEY.ROTATION: quaternion.as_rotation_matrix(prior_q).tolist()}
            extrinsics.append({JSON_KEY.KEY: openmvg_view_id, JSON_KEY.VALUE: pose_data})

    return extrinsics


def _export_openmvg_structure(
        kapture_points_3d: Optional[kapture.Points3d],
        kapture_to_openmvg_view_ids: Dict[str, int],
        kapture_observations: Optional[kapture.Observations] = None,
        kapture_keypoints: Optional[kapture.Keypoints] = None,
        keypoints_type: Optional[str] = None,
        kapture_path: Optional[str] = None,
        tar_handlers: Optional[TarCollection] = None,
) -> Optional[List]:
    """
    Exports kapture 3D points, observations and key points

    :param kapture_points_3d: 3D points to export
    :param kapture_to_openmvg_view_ids: kapture to openmvg view identifiers
    :param kapture_observations: kapture observations if any
    :param kapture_keypoints: kapture key points if any
    :param keypoints_type: key points type if any
    :param kapture_path: path to kapture top directory
    :param tar_handlers: list of tar to use to read kapture data

    :return: openmvg structure to be serialized
    """
    # early check
    if kapture_points_3d is None:
        logger.warning('no 3D points to export.')
        return None

    xyz_coordinates = kapture_points_3d[:, 0:3]
    openmvg_structure = []
    # this loop can be very long, lets show some progress
    hide_progress_bars = logger.getEffectiveLevel() > logging.INFO

    for point_idx, coords in enumerate(tqdm(xyz_coordinates, disable=hide_progress_bars)):
        point_3d_structure = {
            JSON_KEY.KEY: point_idx,
            JSON_KEY.VALUE: {
                JSON_KEY.X: coords.tolist(),
                JSON_KEY.OBSERVATIONS: []
            }
        }
        if kapture_observations is not None and point_idx in kapture_observations and \
                keypoints_type is not None and keypoints_type in kapture_observations[point_idx]:
            for kapture_image_name, feature_point_id in kapture_observations[point_idx, keypoints_type]:
                openmvg_view_id = kapture_to_openmvg_view_ids[kapture_image_name]
                point_2d_observation = {JSON_KEY.KEY: openmvg_view_id,
                                        JSON_KEY.VALUE: {JSON_KEY.ID_FEAT: feature_point_id, }}

                if kapture_path and kapture_keypoints is not None:
                    # if given, load keypoints to populate 2D coordinates of the feature.
                    keypoints_file_path = get_keypoints_fullpath(keypoints_type,
                                                                 kapture_path,
                                                                 kapture_image_name,
                                                                 tar_handlers)
                    try:
                        keypoints_data = image_keypoints_from_file(keypoints_file_path,
                                                                   kapture_keypoints.dtype,
                                                                   kapture_keypoints.dsize)
                        point_2d_observation[JSON_KEY.VALUE][JSON_KEY.x] = \
                            keypoints_data[feature_point_id, 0:2].tolist()
                    except FileNotFoundError:
                        logger.warning(f'unable to load keypoints file {keypoints_file_path}')

                point_3d_structure[JSON_KEY.VALUE][JSON_KEY.OBSERVATIONS].append(point_2d_observation)

        openmvg_structure.append(point_3d_structure)

    return openmvg_structure


def _export_openmvg_sfm_data(
    kapture_data: kapture.Kapture,
    kapture_path: str,
    tar_handlers: TarCollection,
    keypoints_type: Optional[str],
    openmvg_sfm_data_file_path: str,
    openmvg_image_root_path: str,
    image_action: TransferAction,
    image_path_flatten: bool,
    force: bool,
    kapture_to_openmvg_view_ids: Dict
) -> None:
    """
    Convert the kapture data into an openMVG dataset stored as a dictionary, and store it as a json file.
    The format is defined here:
    https://openmvg.readthedocs.io/en/latest/software/SfM/SfM_OutputFormat/

    :param kapture_data: the kapture data
    :param kapture_path: top directory of the kapture data and the images
    :param tar_handlers: tar handlers to use to read the data
    :param keypoints_type: type of key points if any
    :param openmvg_sfm_data_file_path: path to the SfM data file to be written.
    :param openmvg_image_root_path: input path to openMVG image directory to be created.
    :param image_action: action to apply on images: link, copy, move or do nothing.
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :param force: if true, will remove existing openMVG data without prompting the user.
    :param kapture_to_openmvg_view_ids: input/output mapping of kapture image name to corresponding openmvg view id.
    """

    if kapture_data.cameras is None or kapture_data.records_camera is None:
        raise ValueError('export_openmvg_sfm_data needs kapture camera and records_camera.')
    cameras = kapture_data.cameras

    # refer to the original image dir when skipping image transfer.
    if image_action == TransferAction.skip:
        openmvg_image_root_path = get_image_fullpath(kapture_path)

    if openmvg_image_root_path is None:
        raise ValueError(f'openmvg_image_root_path must be defined to be able to perform {image_action}.')

    # make sure directory is ready to contain openmvg_sfm_data_file_path
    os.makedirs(path.dirname(openmvg_sfm_data_file_path), exist_ok=True)

    # Check we don't have other sensors defined
    if len(kapture_data.sensors) != len(kapture_data.cameras):
        extra_sensor_number = len(kapture_data.sensors) - len(kapture_data.cameras)
        logger.warning(f'We will ignore {extra_sensor_number} sensors that are not camera')

    # openmvg does not support rigs
    if kapture_data.rigs:
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

    # polymorphic_status = PolymorphicStatus({}, 1, 1)
    polymorphic_registry = CerealPointerRegistry(id_key=JSON_KEY.POLYMORPHIC_ID, value_key=JSON_KEY.POLYMORPHIC_NAME)
    ptr_wrapper_registry = CerealPointerRegistry(id_key=JSON_KEY.ID, value_key=JSON_KEY.DATA)
    # Compute openMVG identifiers
    kapture_images_data: List[Tuple[int, str, str]] = []  # List of (timestamp, kapture_cam_id, image_name)
    sub_root_path: str = ''
    image_dirs: Set[str] = set()  # all images directories
    kapture_to_openmvg_cam_ids: Dict[str:int] = {}  # kapture_cam_id -> openmvg_cam_id
    for timestamp, image_data in kapture_data.records_camera.items():
        for kapture_cam_id, kapture_image_name in image_data.items():
            kapture_images_data.append((timestamp, kapture_cam_id, kapture_image_name))
            image_dirs.add(path.dirname(kapture_image_name))
            _compute_openmvg_id(kapture_cam_id, kapture_to_openmvg_cam_ids)
            _compute_openmvg_id(kapture_image_name, kapture_to_openmvg_view_ids)
    if len(image_dirs) > 1:
        # Find if they share a top path
        sub_root_path = path.commonpath(list(image_dirs))
    elif len(image_dirs) == 1:
        sub_root_path = image_dirs.pop()
    if sub_root_path:
        openmvg_image_root_path = path.abspath(path.join(openmvg_image_root_path, sub_root_path))
    if image_action == TransferAction.root_link:
        if not sub_root_path:
            # We can not link directly to the top destination openmvg directory
            # We need an additional level
            openmvg_image_root_path = path.join(openmvg_image_root_path, 'images')
        kapture_records_path = get_image_fullpath(kapture_path, sub_root_path)
        # Do a unique images directory link
        # openmvg_root_path -> kapture/<records_dir>/openmvg_top_images_directory
        # beware that the paths are reverted in the symlink call
        os.symlink(kapture_records_path, openmvg_image_root_path)

    logger.debug('exporting intrinsics ...')
    openmvg_sfm_data_intrinsics = _export_openmvg_intrinsics(
        cameras,
        kapture_to_openmvg_cam_ids,
        polymorphic_registry,
        ptr_wrapper_registry
    )

    logger.debug('exporting views ...')
    openmvg_sfm_data_views = _export_openmvg_views(
        cameras,
        kapture_images_data,
        kapture_data.trajectories,
        kapture_to_openmvg_cam_ids,
        kapture_to_openmvg_view_ids,
        polymorphic_registry,
        ptr_wrapper_registry,
        sub_root_path,
        image_path_flatten
    )
    logger.debug('exporting extrinsics ...')
    openmvg_sfm_data_poses = _export_openmvg_extrinsics(
        kapture_images_data,
        kapture_data.trajectories,
        kapture_to_openmvg_view_ids)

    # structure : correspond to kapture observations + 3D points
    logger.debug('exporting structure ...')
    kapture_keypoints = kapture_data.keypoints[keypoints_type] if (
            kapture_data.keypoints is not None and
            keypoints_type is not None and
            keypoints_type in kapture_data.keypoints) else None
    openmvg_sfm_data_structure = _export_openmvg_structure(
        kapture_data.points3d,
        kapture_to_openmvg_view_ids,
        kapture_data.observations,
        kapture_keypoints,
        keypoints_type,
        kapture_path,
        tar_handlers
    )

    openmvg_sfm_data = {
        JSON_KEY.SFM_DATA_VERSION: OPENMVG_SFM_DATA_VERSION_NUMBER,
        JSON_KEY.ROOT_PATH: path.abspath(openmvg_image_root_path),
        JSON_KEY.INTRINSICS: openmvg_sfm_data_intrinsics,
        JSON_KEY.VIEWS: openmvg_sfm_data_views,
        JSON_KEY.EXTRINSICS: openmvg_sfm_data_poses,
        JSON_KEY.STRUCTURE: openmvg_sfm_data_structure,
        JSON_KEY.CONTROL_POINTS: [],
    }

    logger.debug(f'Saving to openmvg {openmvg_sfm_data_file_path}...')
    with open(openmvg_sfm_data_file_path, "w") as fid:
        json.dump(openmvg_sfm_data, fid, indent=4)

    # do the actual image transfer
    if not image_action == TransferAction.skip:
        job_copy = (
            (  # source path -> destination path
                get_image_fullpath(kapture_path, kapture_image_name),
                path.join(openmvg_image_root_path,
                          _get_openmvg_image_path(path.relpath(kapture_image_name, sub_root_path) if sub_root_path
                                                  else kapture_image_name,
                                                  image_path_flatten))
            )
            for kapture_image_name in kapture_data.records_camera.data_list()
        )
        source_filepath_list, destination_filepath_list = zip(*job_copy)
        transfer_files_from_dir(source_filepath_list, destination_filepath_list, image_action, force)


def _export_openmvg_regions(
        kapture_path: str,
        kapture_keypoints: Optional[kapture.Keypoints],
        keypoints_type: Optional[str],
        kapture_descriptors: Optional[kapture.Descriptors],
        descriptors_type: Optional[str],
        tar_handlers: TarCollection,
        openmvg_regions_dir_path: str,
        image_path_flatten: bool
):
    """
    exports openMVG regions ie keypoints and descriptors.

    :param kapture_path: input path to root kapture directory.
    :param kapture_keypoints: input kapture keypoints. Could be None if no keypoints.
    :param keypoints_type: type of key points if any
    :param kapture_descriptors: input kapture descriptors. Could be None if no descriptors.
    :param descriptors_type: type of descriptors if any
    :param tar_handlers: tar handlers to read the data
    :param openmvg_regions_dir_path: input path to output openMVG regions directory.
    :param image_path_flatten: if true, it means that image path are to be flatten.
    """
    # early check we should do
    if kapture_keypoints is None or kapture_descriptors is None or keypoints_type is None or descriptors_type is None:
        logger.warning('no keypoints or descriptors to export.')
        return

    # make sure output directory is ready
    os.makedirs(openmvg_regions_dir_path, exist_ok=True)

    # only able to export SIFT
    if any([f.type_name.upper() != 'SIFT' for f in [kapture_keypoints, kapture_descriptors]]):
        raise ValueError(f'unable to export other regions than sift '
                         f'(got {kapture_keypoints.type_name}/{kapture_descriptors.type_name})')

    os.makedirs(openmvg_regions_dir_path, exist_ok=True)
    polymorphic_registry = CerealPointerRegistry(id_key=JSON_KEY.POLYMORPHIC_ID, value_key=JSON_KEY.POLYMORPHIC_NAME)
    # create image_describer.json
    image_describer_props = {JSON_KEY.PTR_WRAPPER: {JSON_KEY.VALID: 1,
                                                    JSON_KEY.DATA: {JSON_KEY.PARAMS: {
                                                        "root_sift": True
                                                    }
                                                    },
                                                    "bOrientation": True
                                                    }
                             }
    image_describer_props.update(polymorphic_registry.get_ids_dict('SIFT_Image_describer'))
    fake_regions_type = {JSON_KEY.PTR_WRAPPER: {JSON_KEY.VALID: 1,
                                                JSON_KEY.DATA: {JSON_KEY.VALUE0: [],
                                                                JSON_KEY.VALUE1: []
                                                                }
                                                }
                         }
    fake_regions_type.update(polymorphic_registry.get_ids_dict('SIFT_Regions'))
    image_describer = {
        JSON_KEY.IMAGE_DESCRIBER: image_describer_props,
        JSON_KEY.REGIONS_TYPE: fake_regions_type
    }
    image_describer_file_path = path.join(openmvg_regions_dir_path, OPENMVG_DEFAULT_REGIONS_FILE_NAME)
    with open(image_describer_file_path, 'w') as fid:
        json.dump(image_describer, fid, indent=4)

    # this loop can be very long, lets show some progress
    hide_progress_bars = logger.getEffectiveLevel() > logging.INFO

    # copy keypoints files
    keypoints = keypoints_to_filepaths(kapture_keypoints, keypoints_type, kapture_path, tar_handlers)
    for kapture_image_name, kapture_keypoint_file_path in tqdm(keypoints.items(), disable=hide_progress_bars):
        openmvg_keypoint_file_name = _get_openmvg_image_path(kapture_image_name, image_path_flatten)
        openmvg_keypoint_file_name = path.splitext(path.basename(openmvg_keypoint_file_name))[0] + '.feat'
        openmvg_keypoint_file_path = path.join(openmvg_regions_dir_path, openmvg_keypoint_file_name)
        keypoints_data = image_keypoints_from_file(kapture_keypoint_file_path,
                                                   kapture_keypoints.dtype,
                                                   kapture_keypoints.dsize)
        keypoints_data = keypoints_data[:, 0:4]
        np.savetxt(openmvg_keypoint_file_path, keypoints_data, fmt='%10.5f')

    # copy descriptors files
    """
    from openMVG regions_factory.hpp
    using SIFT_Regions = Scalar_Regions<SIOPointFeature, unsigned char, 128>;
    using AKAZE_Float_Regions = Scalar_Regions<SIOPointFeature, float, 64>;
    using AKAZE_Liop_Regions = Scalar_Regions<SIOPointFeature, unsigned char, 144>;
    using AKAZE_Binary_Regions = Binary_Regions<SIOPointFeature, 64>;
    """
    descriptors = descriptors_to_filepaths(kapture_descriptors, descriptors_type, kapture_path, tar_handlers)
    for kapture_image_name, kapture_descriptors_file_path in tqdm(descriptors.items(), disable=hide_progress_bars):
        openmvg_descriptors_file_name = _get_openmvg_image_path(kapture_image_name, image_path_flatten)
        openmvg_descriptors_file_name = path.splitext(path.basename(openmvg_descriptors_file_name))[0] + '.desc'
        openmvg_descriptors_file_path = path.join(openmvg_regions_dir_path, openmvg_descriptors_file_name)
        kapture_descriptors_data = image_descriptors_from_file(kapture_descriptors_file_path,
                                                               kapture_descriptors.dtype,
                                                               kapture_descriptors.dsize)
        # assign a byte array of [size_t[1] + uint8[nb features x 128]
        size_t_len = OPENMVG_DESC_HEADER_BYTES_NUMBER
        openmvg_descriptors_data = np.empty(dtype=np.uint8, shape=(kapture_descriptors_data.size + size_t_len,))
        openmvg_descriptors_data[0:size_t_len] \
            .view(dtype=OPENMVG_DESC_HEADER_DTYPE)[0] = kapture_descriptors_data.shape[0]
        openmvg_descriptors_data[size_t_len:] = kapture_descriptors_data.flatten()
        array_to_file(openmvg_descriptors_file_path, openmvg_descriptors_data)


def _export_openmvg_matches(
        kapture_path: str,
        kapture_data: kapture.Kapture,
        keypoints_type: Optional[str],
        tar_handlers: TarCollection,
        openmvg_matches_file_path: str,
        kapture_to_openmvg_view_ids: Dict[str, int]
):
    if kapture_data.matches is None or keypoints_type is None or keypoints_type not in kapture_data.matches:
        logger.warning('No matches to be exported.')
        return

    if path.splitext(openmvg_matches_file_path)[1] != '.txt':
        logger.warning('Matches are exported as text format, even if file does not ends with .txt.')

    # make sure output directory is ready
    os.makedirs(path.dirname(openmvg_matches_file_path), exist_ok=True)

    hide_progress_bars = logger.getEffectiveLevel() > logging.INFO
    matches = matches_to_filepaths(kapture_data.matches[keypoints_type], keypoints_type, kapture_path, tar_handlers)
    with open(openmvg_matches_file_path, 'w') as fid:
        for image_pair, kapture_matches_filepath in tqdm(matches.items(), disable=hide_progress_bars):
            # idx image1 idx image 2
            # nb pairs
            # pl1 pr1 pl2 pr2 ...
            i, j = [kapture_to_openmvg_view_ids[image_name] for image_name in image_pair]
            fid.write(f'{i} {j}\n')
            matches_indices = image_matches_from_file(kapture_matches_filepath)[:, 0:2].astype(int)
            fid.write(f'{matches_indices.shape[0]}\n')
            for indices_pair in matches_indices:
                fid.write(f'{indices_pair[0]}  {indices_pair[1]}\n')


def export_openmvg(
    kapture_path: str,
    openmvg_sfm_data_file_path: str,
    openmvg_image_root_path: str = None,
    openmvg_regions_dir_path: str = None,
    openmvg_matches_file_path: str = None,
    image_action: TransferAction = TransferAction.skip,
    image_path_flatten: bool = False,
    keypoints_type: Optional[str] = None,
    descriptors_type: Optional[str] = None,
    force: bool = False
) -> None:
    """
    Export the kapture data to an openMVG files.
    If the openmvg_path is a directory, it will create a JSON file (using the default name sfm_data.json)
    in that directory.

    :param kapture_path: full path to input kapture directory
    :param openmvg_sfm_data_file_path: input path to the SfM data file to be written.
    :param openmvg_image_root_path: optional input path to openMVG image directory to be created.
    :param openmvg_regions_dir_path: optional input path to openMVG regions (feat, desc) directory to be created.
    :param openmvg_matches_file_path: optional input path to openMVG matches file to be created.
    :param image_action: an action to apply on the images: relative linking, absolute linking, copy or move. Or top
     directory linking or skip to do nothing. If not "skip" requires openmvg_image_root_path to be defined.
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :param force: if true, will remove existing openMVG data without prompting the user.
    :param keypoints_type: key points type if any
    :param descriptors_type: descriptors type if any
    """

    if any(arg is not None and not isinstance(arg, str)
           for arg in [kapture_path, openmvg_image_root_path, openmvg_regions_dir_path, openmvg_matches_file_path]
           ):
        raise ValueError('expect str (or None) as path argument.')

    # clean before export
    safe_remove_file(openmvg_sfm_data_file_path, force)
    if path.exists(openmvg_sfm_data_file_path):
        raise ValueError(f'{openmvg_sfm_data_file_path} file already exist')

    # load kapture
    logger.info(f'loading kapture {kapture_path}...')
    with kapture.io.csv.get_all_tar_handlers(kapture_path) as tar_handlers:
        kapture_data = kapture.io.csv.kapture_from_dir(kapture_path, tar_handlers=tar_handlers)
        if kapture_data is None or not isinstance(kapture_data, kapture.Kapture):
            raise ValueError(f'unable to load kapture from {kapture_path}')
        kapture_to_openmvg_view_ids = {}

        if keypoints_type is None:
            keypoints_type: str = try_get_only_key_from_collection(kapture_data.keypoints)
        if descriptors_type is None:
            descriptors_type: str = try_get_only_key_from_collection(kapture_data.descriptors)

        logger.info(f'exporting sfm data to {openmvg_sfm_data_file_path} ...')
        _export_openmvg_sfm_data(
            kapture_data,
            kapture_path,
            tar_handlers,
            keypoints_type,
            openmvg_sfm_data_file_path,
            openmvg_image_root_path,
            image_action,
            image_path_flatten,
            force,
            kapture_to_openmvg_view_ids)

        if openmvg_regions_dir_path is not None:
            try:
                logger.info(f'exporting regions to {openmvg_regions_dir_path} ...')
                kapture_keypoints = kapture_data.keypoints[keypoints_type] if (
                        kapture_data.keypoints is not None and
                        keypoints_type is not None and
                        keypoints_type in kapture_data.keypoints) else None
                kapture_descriptors = kapture_data.descriptors[descriptors_type] if (
                        kapture_data.descriptors is not None and
                        descriptors_type is not None and
                        descriptors_type in kapture_data.descriptors) else None
                _export_openmvg_regions(
                    kapture_path,
                    kapture_keypoints,
                    keypoints_type,
                    kapture_descriptors,
                    descriptors_type,
                    tar_handlers,
                    openmvg_regions_dir_path,
                    image_path_flatten
                )
            except ValueError as e:
                logger.error(e)

        if openmvg_matches_file_path is not None:
            try:
                logger.info(f'exporting matches to {openmvg_matches_file_path} ...')
                _export_openmvg_matches(
                    kapture_path,
                    kapture_data,
                    keypoints_type,
                    tar_handlers,
                    openmvg_matches_file_path,
                    kapture_to_openmvg_view_ids
                )
            except ValueError as e:
                logger.error(e)
