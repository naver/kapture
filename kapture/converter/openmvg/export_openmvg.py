# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Kapture to openmvg export functions.
"""

import json
import logging
import os
import os.path as path
from typing import Dict, List, Optional, Union
import quaternion
import numpy as np
# kapture
import kapture
import kapture.io.csv
from kapture.io.binary import TransferAction, transfer_files_from_dir, array_to_file
from kapture.io.records import get_image_fullpath
from kapture.io.features import keypoints_to_filepaths, image_keypoints_from_file, get_keypoints_fullpath
from kapture.io.features import descriptors_to_filepaths, image_descriptors_from_file
from kapture.io.features import matches_to_filepaths, image_matches_from_file
import kapture.io.structure
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
# local
from .openmvg_commons import JSON_KEY, OPENMVG_SFM_DATA_VERSION_NUMBER, OPENMVG_DEFAULT_JSON_FILE_NAME
from .openmvg_commons import CameraModel

logger = logging.getLogger('openmvg')  # Using global openmvg logger

NEW_ID_MASK = 1 << 31  # 10000000 00000000 00000000 00000000
DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


class CerealPointerRegistry:
    NEW_ID_MASK = 1 << 31  # 10000000 00000000 00000000 00000000
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


def _get_data(camera_params: list) -> Dict:
    # w, h, f, cx, cy
    data: Dict[str, Union[int, float, List[float]]] = {JSON_KEY.WIDTH: int(camera_params[0]),
                                                       JSON_KEY.HEIGHT: int(camera_params[1]),
                                                       JSON_KEY.FOCAL_LENGTH: float(camera_params[2]),
                                                       JSON_KEY.PRINCIPAL_POINT: [float(camera_params[3]),
                                                                                  float(camera_params[4])]}
    return data


def _get_intrinsic_pinhole(camera_params: list) -> Dict:
    # w, h, f, cx, cy
    return _get_data(camera_params)


def _get_intrinsic_pinhole_radial_k1(camera_params: list) -> Dict:
    data = _get_data(camera_params)
    # w, h, f, cx, cy, k
    data[JSON_KEY.DISTO_K1] = [float(camera_params[5])]
    return data


def _get_intrinsic_pinhole_radial_k3(camera_params: list) -> Dict:
    data = _get_data(camera_params)
    # w, h, f, cx, cy, k1, k2, k3
    data[JSON_KEY.DISTO_K3] = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7])]
    return data


def _get_intrinsic_pinhole_brown_t2(camera_params: list) -> Dict:
    # w, h, f, cx, cy, k1, k2, k3, t1, t2
    disto_t2 = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7]),
                float(camera_params[8]), float(camera_params[9])]
    return {JSON_KEY.VALUE0: _get_data(camera_params), JSON_KEY.DISTO_T2: disto_t2}


def _get_intrinsic_fisheye(camera_params: list) -> Dict:
    # w, h, f, cx, cy, k1, k2, k3, k4
    fisheye = [float(camera_params[5]), float(camera_params[6]), float(camera_params[7]), float(camera_params[8])]
    return {JSON_KEY.VALUE0: _get_data(camera_params),
            JSON_KEY.FISHEYE: fisheye}


def get_openmvg_camera_id(kapture_camera_id, kapture_to_openmvg_cam_ids):
    """ return a valid openmvg camera id for the given kapture on.
    It keeps kapture_to_openmvg_cam_ids uptodate, and ensure there is no collision.
    """

    if kapture_camera_id in kapture_to_openmvg_cam_ids:
        # already defined
        return kapture_to_openmvg_cam_ids[kapture_camera_id]

    # its not defined, then, make up one
    last_known_openmvg_camera_id = max(list(kapture_to_openmvg_cam_ids.values()) + [-1])
    assert type(last_known_openmvg_camera_id) == int
    openmvg_camera_id = last_known_openmvg_camera_id + 1
    assert openmvg_camera_id not in kapture_to_openmvg_cam_ids
    kapture_to_openmvg_cam_ids[kapture_camera_id] = openmvg_camera_id
    return openmvg_camera_id


def get_openmvg_image_path(kapture_image_name: str, flatten_path: bool = False):
    """ the openmvg image sub path corresponding to the given kapture one. """
    return kapture_image_name if not flatten_path else kapture_image_name.replace('/', '_')


def export_openmvg_intrinsics(
        kapture_cameras,
        kapture_to_openmvg_cam_ids: Dict[str, int],
        polymorphic_registry: CerealPointerRegistry,
        ptr_wrapper_registry: CerealPointerRegistry,
):
    """
    Exports the given kapture cameras to the openMVG sfm_data structure.
    In openMVGm, cameras are referred as Intrinsics camera internal parameters.

    :param kapture_cameras: input kapture cameras to be exported (even if not used).
    :param kapture_to_openmvg_cam_ids: input/output dict that maps kapture camera ids to openMVG camera ids.
    :param polymorphic_registry: input/output polymorphic IDs status
    :param ptr_wrapper_registry: input/output polymorphic IDs status
    :return:
    """
    openmvg_intrinsics = []
    # kapture_to_openmvg_cam_ids = {}
    # process all cameras
    for kapture_cam_id, kapture_camera in kapture_cameras.items():
        openmvg_camera_id = get_openmvg_camera_id(kapture_cam_id, kapture_to_openmvg_cam_ids)
        kapture_cam_type = kapture_camera.camera_type
        kapture_camera_params = kapture_camera.camera_params
        if kapture_cam_type == kapture.CameraType.SIMPLE_PINHOLE:
            # w, h, f, cx, cy
            opnmvg_cam_type = CameraModel.pinhole
            data = _get_intrinsic_pinhole(kapture_camera_params)
        elif kapture_cam_type == kapture.CameraType.PINHOLE:
            # w, h, f, cx, cy
            opnmvg_cam_type = CameraModel.pinhole
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            (kapture_camera_params[2] + kapture_camera_params[3]) / 2,  # fx+fy/2 as f
                            kapture_camera_params[4], kapture_camera_params[5]]  # cx cy
            data = _get_intrinsic_pinhole(faked_params)
        elif kapture_cam_type == kapture.CameraType.SIMPLE_RADIAL:
            # w, h, f, cx, cy, k
            opnmvg_cam_type = CameraModel.pinhole_radial_k1
            data = _get_intrinsic_pinhole_radial_k1(kapture_camera_params)
        elif kapture_cam_type == kapture.CameraType.RADIAL:
            # w, h, f, cx, cy, k1, k2, k3
            opnmvg_cam_type = CameraModel.pinhole_radial_k3
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            kapture_camera_params[2],  # f
                            kapture_camera_params[3], kapture_camera_params[4],  # cx cy
                            kapture_camera_params[5], kapture_camera_params[6], 0  # k1, k2, k3
                            ]
            data = _get_intrinsic_pinhole_radial_k3(faked_params)
        elif kapture_cam_type == kapture.CameraType.FULL_OPENCV or kapture_cam_type == kapture.CameraType.OPENCV:
            # w, h, f, cx, cy, k1, k2, k3, t1, t2
            opnmvg_cam_type = CameraModel.pinhole_brown_t2
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
            opnmvg_cam_type = CameraModel.fisheye
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
            opnmvg_cam_type = CameraModel.fisheye
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
            opnmvg_cam_type = CameraModel.pinhole_radial_k1
            faked_params = [kapture_camera_params[0], kapture_camera_params[1],  # width height
                            max(kapture_camera_params[0], kapture_camera_params[1]) * DEFAULT_FOCAL_LENGTH_FACTOR,
                            # max(w,h)*1.2 as f
                            int(kapture_camera_params[0] / 2), int(kapture_camera_params[1] / 2),  # cx cy
                            0.0]  # k1
            data = _get_intrinsic_pinhole_radial_k1(faked_params)
        else:
            raise ValueError(f'Camera model {kapture_cam_type.value} not supported')

        intrinsic = {}
        cam_type_poly_id = polymorphic_registry.get_ids_dict(opnmvg_cam_type.name)
        intrinsic.update(cam_type_poly_id)

        # it is assumed that this camera is only encountered once
        # set the first bit of ptr_wrapper_id_current to 1
        data_wrapper = ptr_wrapper_registry.get_ids_dict(data)

        intrinsic[JSON_KEY.PTR_WRAPPER] = data_wrapper
        openmvg_intrinsics.append({JSON_KEY.KEY: openmvg_camera_id, JSON_KEY.VALUE: intrinsic})

    return openmvg_intrinsics


def export_openmvg_views(
        kapture_cameras: kapture.Sensors,
        kapture_images: kapture.RecordsCamera,
        kapture_trajectories: kapture.Trajectories,
        kapture_to_openmvg_cam_ids: Dict[str, int],
        kapture_to_openmvg_view_ids: Dict[str, int],
        polymorphic_registry: CerealPointerRegistry,
        ptr_wrapper_registry: CerealPointerRegistry,
        image_path_flatten: bool,
):
    """

    :param kapture_cameras:
    :param kapture_images:
    :param kapture_trajectories:
    :param kapture_to_openmvg_cam_ids: input dict that maps kapture camera ids to openMVG camera ids.
    :param kapture_to_openmvg_view_ids: input dict that maps kapture image names to openMVG view ids.
    :param polymorphic_registry: input/output polymorphic IDs status
    :param ptr_wrapper_registry: input/output polymorphic IDs status
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :return:
    """
    views = []
    # process all images
    for timestamp, kapture_cam_id, kapture_image_name in kapture.flatten(kapture_images):
        assert kapture_cam_id in kapture_to_openmvg_cam_ids
        assert kapture_image_name in kapture_to_openmvg_view_ids
        openmvg_cam_id = get_openmvg_camera_id(kapture_cam_id, kapture_to_openmvg_cam_ids)
        openmvg_view_id = kapture_to_openmvg_view_ids[kapture_image_name]
        openmvg_image_filepath = get_openmvg_image_path(kapture_image_name, image_path_flatten)
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


def export_openmvg_poses(
        kapture_images: kapture.RecordsCamera,
        kapture_trajectories: kapture.Trajectories,
        kapture_to_openmvg_view_ids: Dict[str, int],
):
    """

    :param kapture_images:
    :param kapture_trajectories:
    :param kapture_to_openmvg_view_ids: input dict that maps kapture image ids to openMVG view ids.
    :return:
    """
    extrinsics = []
    # process all images
    for timestamp, kapture_cam_id, kapture_image_name in kapture.flatten(kapture_images):
        assert kapture_image_name in kapture_to_openmvg_view_ids
        openmvg_view_id = kapture_to_openmvg_view_ids[kapture_image_name]
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


def export_openmvg_structure(
        kapture_points_3d: kapture.Points3d,
        kapture_to_openmvg_view_ids: Dict[str, int],
        kapture_observations: Optional[kapture.Observations],
        kapture_keypoints: Optional[kapture.Keypoints],
        kapture_path: Optional[str],
):
    xyz_coordinates = kapture_points_3d[:, 0:3]
    include_2d_observations = kapture_observations is not None
    openmvg_structure = []
    for point_idx, coords in enumerate(xyz_coordinates):
        point_3d_structure = {
            'key': point_idx,
            'value': {
                'X': coords.tolist(),
                'observations': []
            }
        }
        if include_2d_observations and point_idx in kapture_observations:
            for kapture_image_name, feature_point_id in kapture_observations[point_idx]:
                openmvg_view_id = kapture_to_openmvg_view_ids[kapture_image_name]
                point_2d_observation = {'key': openmvg_view_id,
                                        'value': {'id_feat': feature_point_id, }}

                if kapture_path and kapture_keypoints is not None:
                    # if given, load keypoints to populate 2D coordinates of the feature.
                    keypoints_file_path = get_keypoints_fullpath(kapture_path, kapture_image_name)
                    keypoints_data = image_keypoints_from_file(keypoints_file_path,
                                                               dtype=kapture_keypoints.dtype,
                                                               dsize=kapture_keypoints.dsize)
                    point_2d_observation['value']['x'] = keypoints_data[feature_point_id, 0:2].tolist()

                point_3d_structure['value']['observations'].append(point_2d_observation)

        openmvg_structure.append(point_3d_structure)

    return openmvg_structure


def export_openmvg_sfm_data(
        kapture_path: str,
        kapture_data: kapture.Kapture,
        openmvg_sfm_data_file_path: str,
        openmvg_image_root_path: str,
        image_action: TransferAction,
        image_path_flatten: bool,
        force: bool,
        kapture_to_openmvg_view_ids: dict = {}
) -> Dict:
    """
    Convert the kapture data into an openMVG dataset stored as a dictionary.
    The format is defined here:
    https://openmvg.readthedocs.io/en/latest/software/SfM/SfM_OutputFormat/

    :param kapture_data: the kapture data
    :param kapture_path: top directory of the kapture data and the images
    :param openmvg_sfm_data_file_path: input path to the SfM data file to be written.
    :param openmvg_image_root_path: input path to openMVG image directory to be created.
    :param image_action: action to apply on images: link, copy, move or do nothing.
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :param force: if true, will remove existing openMVG data without prompting the user.
    :param kapture_to_openmvg_view_ids: input/output mapping of kapture image name to corresponding openmvg view id.
    :return: an SfM_data, the openmvg structure, stored as a dictionary ready to be serialized
    """

    if kapture_data.cameras is None or kapture_data.records_camera is None:
        raise ValueError('export_openmvg_sfm_data needs kapture camera and records_camera.')

    if image_action == TransferAction.root_link:
        raise NotImplementedError('root link is not implemented, use skip instead.')

    # refer to the original image dir when skipping image transfer.
    if image_action == TransferAction.skip:
        openmvg_image_root_path = get_image_fullpath(kapture_path)

    if openmvg_image_root_path is None:
        raise ValueError(f'openmvg_image_root_path must be defined to be able to perform {image_action}.')

    # Check we don't have other sensors defined
    if len(kapture_data.sensors) != len(kapture_data.cameras):
        extra_sensor_number = len(kapture_data.sensors) - len(kapture_data.cameras)
        logger.warning(f'We will ignore {extra_sensor_number} sensors that are not camera')

    # openmvg does not support rigs
    if kapture_data.rigs:
        logger.info('remove rigs notation.')
        rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)
        kapture_data.rigs.clear()

    # Compute root path and camera used in records
    kapture_to_openmvg_cam_ids = {}  # kapture_cam_id -> openmvg_cam_id
    for i, (_, _, kapture_image_name) in enumerate(kapture.flatten(kapture_data.records_camera)):
        if kapture_image_name not in kapture_to_openmvg_view_ids:
            kapture_to_openmvg_view_ids[kapture_image_name] = i

    # polymorphic_status = PolymorphicStatus({}, 1, 1)
    polymorphic_registry = CerealPointerRegistry(id_key=JSON_KEY.POLYMORPHIC_ID, value_key=JSON_KEY.POLYMORPHIC_NAME)
    ptr_wrapper_registry = CerealPointerRegistry(id_key=JSON_KEY.ID, value_key=JSON_KEY.DATA)

    openmvg_sfm_data_intrinsics = export_openmvg_intrinsics(
        kapture_cameras=kapture_data.cameras,
        kapture_to_openmvg_cam_ids=kapture_to_openmvg_cam_ids,
        polymorphic_registry=polymorphic_registry,
        ptr_wrapper_registry=ptr_wrapper_registry,
    )

    openmvg_sfm_data_views = export_openmvg_views(
        kapture_cameras=kapture_data.cameras,
        kapture_images=kapture_data.records_camera,
        kapture_trajectories=kapture_data.trajectories,
        kapture_to_openmvg_cam_ids=kapture_to_openmvg_cam_ids,
        kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids,
        polymorphic_registry=polymorphic_registry,
        ptr_wrapper_registry=ptr_wrapper_registry,
        image_path_flatten=image_path_flatten,
    )

    openmvg_sfm_data_poses = export_openmvg_poses(
        kapture_images=kapture_data.records_camera,
        kapture_trajectories=kapture_data.trajectories,
        kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids)

    # structure : correspond to kapture observations + 3D points
    openmvg_sfm_data_structure = export_openmvg_structure(
        kapture_points_3d=kapture_data.points3d,
        kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids,
        kapture_observations=kapture_data.observations,
        kapture_keypoints=kapture_data.keypoints,
        kapture_path=kapture_path
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

    logger.info(f'Saving to openmvg {openmvg_sfm_data_file_path}...')
    with open(openmvg_sfm_data_file_path, "w") as fid:
        json.dump(openmvg_sfm_data, fid, indent=4)

    # do the actual image transfer
    if not image_action == TransferAction.skip:
        job_copy = (
            (  # source path -> dest path
                get_image_fullpath(kapture_path, kapture_image_name),
                path.join(openmvg_image_root_path, get_openmvg_image_path(kapture_image_name, image_path_flatten))
            )
            for _, _, kapture_image_name in kapture.flatten(kapture_data.records_camera)
        )
        source_filepath_list, destination_filepath_list = zip(*job_copy)
        transfer_files_from_dir(
            source_filepath_list=source_filepath_list,
            destination_filepath_list=destination_filepath_list,
            copy_strategy=image_action,
            force_overwrite=force
        )


def export_openmvg_regions(
        kapture_path: str,
        kapture_data: kapture.Kapture,
        openmvg_regions_dir_path: str,
        image_path_flatten: bool
):
    """
    exports openMVG regions ie keypoints and descriptors.

    :param kapture_path:
    :param kapture_data:
    :param openmvg_regions_dir_path:
    :param image_path_flatten:
    :return:
    """
    # only able to export SIFT
    if any([f.type_name.upper() != 'SIFT' for f in [kapture_data.keypoints, kapture_data.descriptors]]):
        raise ValueError(f'unable to export other regions than sift '
                         f'(got {kapture_data.keypoints.type_name}/{kapture_data.descriptors.type_name})')

    os.makedirs(openmvg_regions_dir_path, exist_ok=True)
    polymorphic_registry = CerealPointerRegistry(id_key=JSON_KEY.POLYMORPHIC_ID, value_key=JSON_KEY.POLYMORPHIC_NAME)
    # create image_describer.json
    fake_regions_type = {"ptr_wrapper": {"valid": 1, "data": {"value0": [], "value1": []}}}
    fake_regions_type.update(polymorphic_registry.get_ids_dict('SIFT_Regions'))
    image_describer = {
        'regions_type': fake_regions_type
    }
    image_describer_file_path = path.join(openmvg_regions_dir_path, 'image_describer.json')
    with open(image_describer_file_path, 'w') as fid:
        json.dump(image_describer, fid, indent=4)

    # copy keypoints files
    keypoints = keypoints_to_filepaths(kapture_data.keypoints, kapture_path)
    for kapture_image_name, kapture_keypoint_file_path in keypoints.items():
        openmvg_keypoint_file_name = get_openmvg_image_path(kapture_image_name, image_path_flatten)
        openmvg_keypoint_file_name = path.splitext(path.basename(openmvg_keypoint_file_name))[0] + '.feat'
        openmvg_keypoint_file_path = path.join(openmvg_regions_dir_path, openmvg_keypoint_file_name)
        keypoints_data = image_keypoints_from_file(kapture_keypoint_file_path,
                                                   kapture_data.keypoints.dtype,
                                                   kapture_data.keypoints.dsize)
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
    descriptors = descriptors_to_filepaths(kapture_data.descriptors, kapture_path)
    for kapture_image_name, kapture_descriptors_file_path in descriptors.items():
        openmvg_descriptors_file_name = get_openmvg_image_path(kapture_image_name, image_path_flatten)
        openmvg_descriptors_file_name = path.splitext(path.basename(openmvg_descriptors_file_name))[0] + '.desc'
        openmvg_descriptors_file_path = path.join(openmvg_regions_dir_path, openmvg_descriptors_file_name)
        kapture_descriptors_data = image_descriptors_from_file(kapture_descriptors_file_path,
                                                               kapture_data.descriptors.dtype,
                                                               kapture_data.descriptors.dsize)
        # assign a byte array of [size_t[1] + uint8[nb features x 128]
        size_t_len = 64 // 8
        openmvg_descriptors_data = np.empty(dtype=np.uint8, shape=(kapture_descriptors_data.size + size_t_len,))
        openmvg_descriptors_data[0:size_t_len].view(dtype=np.uint64)[0] = kapture_descriptors_data.shape[0]
        openmvg_descriptors_data[size_t_len:] = kapture_descriptors_data.flatten()
        array_to_file(openmvg_descriptors_file_path, openmvg_descriptors_data)


def export_openmvg_matches(
        kapture_path: str,
        kapture_data: kapture.Kapture,
        openmvg_matches_file_path: str,
        kapture_to_openmvg_view_ids: Dict[str, int]
):
    if kapture_data.matches is None:
        logger.warning('No matches to be exported.')
        return

    if path.splitext(openmvg_matches_file_path)[1] != '.txt':
        logger.warning('Matches are exported as text format, even if file does not ends with .txt.')

    matches = matches_to_filepaths(kapture_data.matches, kapture_path)
    with open(openmvg_matches_file_path, 'w') as fid:
        for image_pair, kapture_matches_filepath in matches.items():
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
     directory linking or skip to do nothing. If not "skip" equires openmvg_image_root_path to be defined.
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :param force: if true, will remove existing openMVG data without prompting the user.
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
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path)
    assert isinstance(kapture_data, kapture.Kapture)
    kapture_to_openmvg_view_ids = {}

    export_openmvg_sfm_data(
        kapture_data=kapture_data,
        kapture_path=kapture_path,
        openmvg_sfm_data_file_path=openmvg_sfm_data_file_path,
        openmvg_image_root_path=openmvg_image_root_path,
        image_action=image_action,
        image_path_flatten=image_path_flatten,
        force=force,
        kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids)

    if openmvg_regions_dir_path is not None:
        try:
            export_openmvg_regions(
                kapture_path=kapture_path,
                kapture_data=kapture_data,
                openmvg_regions_dir_path=openmvg_regions_dir_path,
                image_path_flatten=image_path_flatten
            )
        except ValueError as e:
            logger.error(e)

    if openmvg_matches_file_path is not None:
        try:
            export_openmvg_matches(
                kapture_path=kapture_path,
                kapture_data=kapture_data,
                openmvg_matches_file_path=openmvg_matches_file_path,
                kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids
            )
        except ValueError as e:
            logger.error(e)
