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
from typing import Dict, List, Tuple, Union
import quaternion
from dataclasses import dataclass
# kapture
import kapture
import kapture.io.csv
from kapture.io.binary import TransferAction, transfer_files_from_dir
from kapture.io.records import get_image_fullpath
import kapture.io.structure
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.utils.paths import safe_remove_file, safe_remove_any_path
# local
from .openmvg_commons import JSON_KEY, OPENMVG_SFM_DATA_VERSION_NUMBER, OPENMVG_DEFAULT_JSON_FILE_NAME
from .openmvg_commons import CameraModel

logger = logging.getLogger('openmvg')  # Using global openmvg logger

NEW_ID_MASK = 1 << 31  # 10000000 00000000 00000000 00000000
VIEW_SPECIAL_POLYMORPHIC_ID = 1 << 30  # 01000000 00000000 00000000 00000000
DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


@dataclass
class PolymorphicStatus:
    id_types: dict
    id_cur: int = 1
    ptr_wrapper_id_cur: int = 1


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


def export_openmvg_intrinsics(
        kapture_cameras,
        kapture_to_openmvg_cam_ids: Dict[str, int],
        polymorphic_status: PolymorphicStatus,
):
    """
    Exports the given kapture cameras to the openMVG sfm_data structure.
    In openMVGm, cameras are referred as Intrinsics camera internal parameters.

    :param kapture_cameras: input kapture cameras to be exported (even if not used).
    :param kapture_to_openmvg_cam_ids: input/output dict that maps kapture camera ids to openMVG camera ids.
    :param polymorphic_status: input/output polymorphic IDs status
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
        if opnmvg_cam_type not in polymorphic_status.id_types:
            # if this is the first time model_used is encountered
            # set the first bit of polymorphic_id_current to 1
            intrinsic[JSON_KEY.POLYMORPHIC_ID] = polymorphic_status.id_cur | NEW_ID_MASK
            intrinsic[JSON_KEY.POLYMORPHIC_NAME] = opnmvg_cam_type.name
            polymorphic_status.id_types[opnmvg_cam_type] = polymorphic_status.id_cur
            polymorphic_status.id_cur += 1
        else:
            intrinsic[JSON_KEY.POLYMORPHIC_ID] = polymorphic_status.id_types[opnmvg_cam_type]

        # it is assumed that this camera is only encountered once
        # set the first bit of ptr_wrapper_id_current to 1
        data_wrapper = {JSON_KEY.ID: polymorphic_status.ptr_wrapper_id_cur | NEW_ID_MASK,
                        JSON_KEY.DATA: data}
        polymorphic_status.ptr_wrapper_id_cur += 1

        intrinsic[JSON_KEY.PTR_WRAPPER] = data_wrapper
        openmvg_intrinsics.append({JSON_KEY.KEY: openmvg_camera_id, JSON_KEY.VALUE: intrinsic})

    return openmvg_intrinsics


def export_openmvg_views(
        kapture_cameras: kapture.Sensors,
        kapture_images: kapture.RecordsCamera,
        kapture_trajectories: kapture.Trajectories,
        kapture_to_openmvg_cam_ids: Dict[str, int],
        kapture_to_openmvg_image_paths: Dict[str, str],
        kapture_to_openmvg_view_ids: Dict[str, int],
        polymorphic_status: PolymorphicStatus,
):
    """

    :param kapture_cameras:
    :param kapture_images:
    :param kapture_trajectories:
    :param kapture_to_openmvg_cam_ids: input dict that maps kapture camera ids to openMVG camera ids.
    :param kapture_to_openmvg_image_paths: input dict that maps kapture image path to openMVG image path.
    :param kapture_to_openmvg_view_ids: input dict that maps kapture image names to openMVG view ids.
    :param polymorphic_status: input/output polymorphic IDs status
    :return:
    """
    views = []
    # process all images
    for timestamp, kapture_cam_id, kapture_image_name in kapture.flatten(kapture_images):
        assert kapture_cam_id in kapture_to_openmvg_cam_ids
        assert kapture_image_name in kapture_to_openmvg_view_ids
        assert kapture_image_name in kapture_to_openmvg_image_paths
        openmvg_cam_id = get_openmvg_camera_id(kapture_cam_id, kapture_to_openmvg_cam_ids)
        openmvg_view_id = kapture_to_openmvg_view_ids[kapture_image_name]
        openmvg_image_filepath = kapture_to_openmvg_image_paths[kapture_image_name]
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
            view[JSON_KEY.POLYMORPHIC_ID] = VIEW_SPECIAL_POLYMORPHIC_ID
        else:
            # there is a pose for that timestamp
            # The poses are stored both as priors (in the 'views' table) and as known poses (in the 'extrinsics' table)
            assert kapture_cam_id in kapture_trajectories[timestamp]
            if JSON_KEY.VIEW_PRIORS not in polymorphic_status.id_types:
                # if this is the first time view_priors is encountered
                # set the first bit of polymorphic_id_current to 1
                view[JSON_KEY.POLYMORPHIC_ID] = polymorphic_status.id_cur | NEW_ID_MASK
                view[JSON_KEY.POLYMORPHIC_NAME] = JSON_KEY.VIEW_PRIORS
                polymorphic_status.id_types[JSON_KEY.VIEW_PRIORS] = polymorphic_status.id_cur
                polymorphic_status.id_cur += 1
            else:
                view[JSON_KEY.POLYMORPHIC_ID] = polymorphic_status.id_types[JSON_KEY.VIEW_PRIORS]

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
        # set the first bit of ptr_wrapper_id_current to 1
        view_wrapper = {JSON_KEY.ID: polymorphic_status.ptr_wrapper_id_cur | NEW_ID_MASK,
                        JSON_KEY.DATA: view_data}
        polymorphic_status.ptr_wrapper_id_cur += 1

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


def export_openmvg_sfm_data(
        kapture_path: str,
        kapture_data: kapture.Kapture,
        openmvg_sfm_data_file_path: str,
        openmvg_image_root_path: str,
        image_path_flatten: bool,
        image_action: TransferAction,
        force: bool
) -> Dict:
    """
    Convert the kapture data into an openMVG dataset stored as a dictionary.
    The format is defined here:
    https://openmvg.readthedocs.io/en/latest/software/SfM/SfM_OutputFormat/

    :param kapture_data: the kapture data
    :param kapture_path: top directory of the kapture data and the images
    :param openmvg_sfm_data_file_path: input path to the SfM data file to be written.
    :param openmvg_image_root_path: input path to openMVG image directory to be created.
    :param image_path_flatten: flatten image path (eg. to avoid image name collision in openMVG regions).
    :param image_action: action to apply on images: link, copy, move or do nothing.
    :return: an SfM_data, the openmvg structure, stored as a dictionary ready to be serialized
    """

    if kapture_data.cameras is None or kapture_data.records_camera is None:
        raise ValueError('export_openmvg_sfm_data needs kapture camera and records_camera.')

    if image_action == TransferAction.root_link:
        raise NotImplementedError('root link is not implemented, use skip instead.')

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
    kapture_to_openmvg_view_ids = {
        kapture_image_name: i
        for i, (_, _, kapture_image_name) in enumerate(kapture.flatten(kapture_data.records_camera))
    }
    kapture_to_openmvg_image_paths = {  # kapture_image_name -> openmvg_image_path
        kapture_image_name: kapture_image_name
        for _, _, kapture_image_name in kapture.flatten(kapture_data.records_camera)
    }

    if image_path_flatten:
        if image_action == TransferAction.skip or image_action == TransferAction.root_link:
            raise ValueError(f'cant flatten image filename with transfer {image_action}')
        kapture_to_openmvg_image_paths = {
            k: v.replace('/', '_')
            for k, v in kapture_to_openmvg_image_paths.items()
        }

    polymorphic_status = PolymorphicStatus({}, 1, 1)

    openmvg_sfm_data_intrinsics = export_openmvg_intrinsics(
        kapture_cameras=kapture_data.cameras,
        kapture_to_openmvg_cam_ids=kapture_to_openmvg_cam_ids,
        polymorphic_status=polymorphic_status)

    openmvg_sfm_data_views = export_openmvg_views(
        kapture_cameras=kapture_data.cameras,
        kapture_images=kapture_data.records_camera,
        kapture_trajectories=kapture_data.trajectories,
        kapture_to_openmvg_cam_ids=kapture_to_openmvg_cam_ids,
        kapture_to_openmvg_image_paths=kapture_to_openmvg_image_paths,
        kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids,
        polymorphic_status=polymorphic_status)

    openmvg_sfm_data_poses = export_openmvg_poses(
        kapture_images=kapture_data.records_camera,
        kapture_trajectories=kapture_data.trajectories,
        kapture_to_openmvg_view_ids=kapture_to_openmvg_view_ids)

    openmvg_sfm_data = {
        JSON_KEY.SFM_DATA_VERSION: OPENMVG_SFM_DATA_VERSION_NUMBER,
        JSON_KEY.ROOT_PATH: path.abspath(openmvg_image_root_path),
        JSON_KEY.INTRINSICS: openmvg_sfm_data_intrinsics,
        JSON_KEY.VIEWS: openmvg_sfm_data_views,
        JSON_KEY.EXTRINSICS: openmvg_sfm_data_poses,
        JSON_KEY.STRUCTURE: [],
        JSON_KEY.CONTROL_POINTS: [],
    }

    logger.info(f'Saving to openmvg {openmvg_sfm_data_file_path}...')
    with open(openmvg_sfm_data_file_path, "w") as fid:
        json.dump(openmvg_sfm_data, fid, indent=4)

    # do the actual image transfer
    if not image_action == TransferAction.skip:
        job_copy = (
            (get_image_fullpath(kapture_path, kapture_image_name), path.join(openmvg_image_root_path, openmvg_image_path))
            for kapture_image_name, openmvg_image_path in kapture_to_openmvg_image_paths.items()
        )
        source_filepath_list, destination_filepath_list = zip(*job_copy)
        transfer_files_from_dir(
            source_filepath_list=source_filepath_list,
            destination_filepath_list=destination_filepath_list,
            copy_strategy=image_action,
            force_overwrite=force
        )


def export_openmvg(
        kapture_path: str,
        openmvg_sfm_data_file_path: str,
        openmvg_image_root_path: str,
        openmvg_regions_dir_path: str,
        openmvg_matches_file_path: str,
        image_action: TransferAction,
        force: bool = False
) -> None:
    """
    Export the kapture data to an openMVG files.
    If the openmvg_path is a directory, it will create a JSON file (using the default name sfm_data.json)
    in that directory.

    :param kapture_path: full path to the top kapture directory
    :param openmvg_sfm_data_file_path: input path to the SfM data file to be written.
    :param openmvg_image_root_path: input path to openMVG image directory to be created.
    :param openmvg_regions_dir_path: optional input path to openMVG regions (feat, desc) directory to be created.
    :param openmvg_matches_file_path: optional input path to openMVG matches file to be created.
    :param image_action: an action to apply on the images: relative linking, absolute linking, copy or move. Or top
     directory linking or skip to do nothing.
    :param force: if true, will remove existing openMVG data without prompting the user.
    """

    # clean before export
    safe_remove_file(openmvg_sfm_data_file_path, force)
    if path.exists(openmvg_sfm_data_file_path):
        raise ValueError(f'{openmvg_sfm_data_file_path} file already exist')

    # TODO: move this in export_openmvg_sfm_data
    if image_action == TransferAction.skip:
        openmvg_image_root_path = get_image_fullpath(kapture_path)

    if image_action != TransferAction.skip:
        if path.isdir(openmvg_image_root_path):
            safe_remove_any_path(openmvg_image_root_path, force)
            if path.isdir(openmvg_image_root_path):
                raise ValueError(f'Images directory {openmvg_image_root_path} exist with remaining files')
        os.makedirs(openmvg_image_root_path, exist_ok=True)

    # load kapture
    logger.info(f'loading kapture {kapture_path}...')
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_path)
    assert isinstance(kapture_data, kapture.Kapture)

    export_openmvg_sfm_data(
        kapture_data=kapture_data,
        kapture_path=kapture_path,
        openmvg_sfm_data_file_path=openmvg_sfm_data_file_path,
        openmvg_image_root_path=openmvg_image_root_path,
        image_action=image_action,
        image_path_flatten=False,
        force=force)
