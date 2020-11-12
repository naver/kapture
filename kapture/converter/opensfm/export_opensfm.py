# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Kapture to opensfm export functions.
"""

import os
import logging
import os.path as path
import numpy as np
import quaternion
import gzip
import pickle
import json
from tqdm import tqdm
from typing import Any, Dict
# kapture
import kapture
from kapture.io.binary import TransferAction, transfer_files_from_dir
from kapture.io.features import get_keypoints_fullpath, image_keypoints_from_file
from kapture.io.features import get_descriptors_fullpath, image_descriptors_from_file
from kapture.io.features import get_matches_fullpath, image_matches_from_file
from kapture.io.records import get_record_fullpath


logger = logging.getLogger('opensfm')


"""
opensfm_project/
├── config.yaml
├── images/
├── masks/
├── gcp_list.txt
├── exif/
├── camera_models.json
├── features/
├── matches/
├── tracks.csv
├── reconstruction.json
├── reconstruction.meshed.json
└── undistorted/
    ├── images/
    ├── masks/
    ├── tracks.csv
    ├── reconstruction.json
    └── depthmaps/
        └── merged.ply
"""

"""
reconstruction.json: [RECONSTRUCTION, ...]

RECONSTRUCTION: {
    "cameras": {
        CAMERA_ID: CAMERA,
        ...
    },
    "shots": {
        SHOT_ID: SHOT,
        ...
    },
    "points": {
        POINT_ID: POINT,
        ...
    }
}

CAMERA: {
    "projection_type": "perspective",  # Can be perspective, brown, fisheye or equirectangular
    "width": NUMBER,                   # Image width in pixels
    "height": NUMBER,                  # Image height in pixels

    # Depending on the projection type more parameters are stored.
    # These are the parameters of the perspective camera.
    "focal": NUMBER,                   # Estimated focal length
    "k1": NUMBER,                      # Estimated distortion coefficient
    "k2": NUMBER,                      # Estimated distortion coefficient
}

SHOT: {
    "camera": CAMERA_ID,
    "rotation": [X, Y, Z],      # Estimated rotation as an angle-axis vector
    "translation": [X, Y, Z],   # Estimated translation
    "gps_position": [X, Y, Z],  # GPS coordinates in the reconstruction reference frame
    "gps_dop": METERS,          # GPS accuracy in meters
    "orientation": NUMBER,      # EXIF orientation tag (can be 1, 3, 6 or 8)
    "capture_time": SECONDS     # Capture time as a UNIX timestamp
}

POINT: {
    "coordinates": [X, Y, Z],      # Estimated position of the point
    "color": [R, G, B],            # Color of the point
}
"""

"""
reconstruction.meshed.json
[{
    'cameras': {'v2 unknown unknown 1920 1080 perspective 0':
                {'projection_type': 'perspective', 'width': 1920, 'height': 1080, 'focal': 0.8647151305270488,
                'k1': 0.04060214391621549, 'k2': -0.04060273810852096}},
    'shots': {
        'frame00016.png' : {
        'rotation': [1.5061234719524716, 0.06688721174244067, -0.030847050348337724],
         'translation': [-2.1535823328020456, 0.28345212194377944, 1.2491740134158436],
         'camera': 'v2 unknown unknown 1920 1080 perspective 0',
         'orientation': 1,
         'capture_time': 0.0,
         'gps_dop', 'gps_position',
         'vertices',     # only in meshed
         'faces' [[x, y, z, ...]  # only in meshed
         },
         ...
    },
     'points': {
        '1': {
            'color': [74.0, 43.0, 31.0],
            'coordinates': [-4.204454761953588, 11.796709404713068, 4.7276044200915]
        },
        ...
     },
     'reference_lla': {
        'latitude': 0.0,
        'longitude': 0.0,
        'altitude': 0
     }
}]
"""


def export_opensfm_camera(
        kapture_camera: kapture.Camera
) -> Dict[str, Any]:
    """
    Converts kapture camera to OpenSfM.
    OpenSfm propose 3 models of camera: perspective, fisheye, spherical.
    Perspective Camera of OpenSfM
    (see https://www.opensfm.org/docs/geometry.html#camera-models)

    OpenSfM camera axis:
        - The z-axis points forward
        - The y-axis points down
        - The x-axis points to the right

    OpenSfM Pose:
        OpenSfM, however, chooses not to store the “camera origin” in Pose objects.
        Instead, it stores the camera coordinates of the world origin in the translation field.
        Meaning Camera from World transformation.
        - rotation is represented as axis-angle vector.

    :param kapture_camera: camera kapture definition
    :return: camera definition as dictionary to save (as json, db record, ...)
    """
    corresponding_opensfm_perspective = [
        kapture.CameraType.SIMPLE_PINHOLE,
        kapture.CameraType.RADIAL,
        kapture.CameraType.SIMPLE_RADIAL
    ]

    if kapture_camera.camera_type not in corresponding_opensfm_perspective:
        raise ValueError(f'Unable to export camera of type "{kapture_camera.camera_type}" to OpenSfM')

    # SIMPLE_PINHOLE = w, h, fx, fy, cx, cy
    # SIMPLE_RADIAL  = w, h, f, cx, cy, k1
    # RADIAL         = w, h, f, cx, cy, k1, k2

    # note that principal point is ignored by openSfM
    largest_side_in_pixels = max(kapture_camera.camera_params[0], kapture_camera.camera_params[1])
    opensfm_camera = {
        'projection_type': 'perspective',
        'width': int(kapture_camera.camera_params[0]),
        'height': int(kapture_camera.camera_params[1]),
        # The focal length provided by the EXIF metadata divided by the sensor width
        'focal': kapture_camera.camera_params[2] / largest_side_in_pixels,
        'k1': 0.0,
        'k2': 0.0,
    }

    # Distortion model of OpenSfM match CameraType.RADIAL
    if kapture_camera.camera_type in [kapture.CameraType.SIMPLE_RADIAL, kapture.CameraType.RADIAL]:
        opensfm_camera['k1'] = kapture_camera.camera_params[5]

    if kapture_camera.camera_type == kapture.CameraType.RADIAL:
        opensfm_camera['k2'] = kapture_camera.camera_params[6]

    return opensfm_camera


def _export_opensfm_features_and_matches(image_filenames, kapture_data, kapture_root_dir, opensfm_root_dir,
                                         disable_tqdm):
    """
    export features files (keypoints + descriptors) and matches
    """
    opensfm_features_suffix = '.features.npz'
    opensfm_features_dir_path = path.join(opensfm_root_dir, 'features')
    logger.info(f'exporting keypoint and descriptors to {opensfm_features_dir_path}')
    os.makedirs(opensfm_features_dir_path, exist_ok=True)
    for image_filename in tqdm(image_filenames, disable=disable_tqdm):
        opensfm_features = {}
        # look and load for keypoints in kapture
        if kapture_data.keypoints is not None and image_filename in kapture_data.keypoints:
            kapture_keypoints_filepath = get_keypoints_fullpath(kapture_dirpath=kapture_root_dir,
                                                                image_filename=image_filename)
            logger.debug(f'loading {kapture_keypoints_filepath}')
            kapture_keypoint = image_keypoints_from_file(kapture_keypoints_filepath,
                                                         dtype=kapture_data.keypoints.dtype,
                                                         dsize=kapture_data.keypoints.dsize)
            opensfm_features['points'] = kapture_keypoint

        # look and load for descriptors in kapture
        if kapture_data.descriptors is not None and image_filename in kapture_data.descriptors:
            kapture_descriptor_filepath = get_descriptors_fullpath(kapture_dirpath=kapture_root_dir,
                                                                   image_filename=image_filename)
            logger.debug(f'loading {kapture_descriptor_filepath}')
            kapture_descriptor = image_descriptors_from_file(kapture_descriptor_filepath,
                                                             dtype=kapture_data.descriptors.dtype,
                                                             dsize=kapture_data.descriptors.dsize)
            opensfm_features['descriptors'] = kapture_descriptor

        # writing opensfm feature file
        if len(opensfm_features) > 0:
            opensfm_features_filepath = path.join(opensfm_features_dir_path, image_filename + opensfm_features_suffix)
            logger.debug(f'writing {opensfm_features_filepath}')
            os.makedirs(path.dirname(opensfm_features_filepath), exist_ok=True)
            np.save(opensfm_features_filepath, opensfm_features)

    # export matches files
    if kapture_data.matches is not None:
        opensfm_matches_suffix = '_matches.pkl.gz'
        opensfm_matches_dir_path = path.join(opensfm_root_dir, 'matches')
        os.makedirs(opensfm_matches_dir_path, exist_ok=True)
        logger.info(f'exporting matches to {opensfm_matches_dir_path}')
        opensfm_pairs = {}
        for image_filename1, image_filename2 in kapture_data.matches:
            opensfm_pairs.setdefault(image_filename1, []).append(image_filename2)

        for image_filename1 in tqdm(image_filenames, disable=disable_tqdm):
            opensfm_matches = {}
            opensfm_matches_filepath = path.join(opensfm_matches_dir_path, image_filename1 + opensfm_matches_suffix)
            logger.debug(f'loading matches for {image_filename1}')
            for image_filename2 in opensfm_pairs.get(image_filename1, []):
                # print(image_filename1, image_filename2)
                kapture_matches_filepath = get_matches_fullpath(
                    (image_filename1, image_filename2), kapture_dirpath=kapture_root_dir)
                kapture_matches = image_matches_from_file(kapture_matches_filepath)
                opensfm_matches[image_filename2] = kapture_matches[:, 0:2].astype(np.int)

            os.makedirs(path.dirname(opensfm_matches_filepath), exist_ok=True)
            with gzip.open(opensfm_matches_filepath, 'wb') as f:
                pickle.dump(opensfm_matches, f)


def export_opensfm(
        kapture_root_dir: str,
        opensfm_root_dir: str,
        force_overwrite_existing: bool = False,
        images_export_method: TransferAction = TransferAction.copy
) -> None:
    """
    Export the kapture data to an openSfM format

    :param kapture_root_dir: full path to the top kapture directory
    :param opensfm_root_dir: path of the directory where to store the data in openSfM format
    :param force_overwrite_existing: if true, will remove existing openSfM data without prompting the user.
    :param images_export_method:
    """

    disable_tqdm = logger.getEffectiveLevel() > logging.INFO  # don't display tqdm for non-verbose levels
    # load reconstruction
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_root_dir)

    # export cameras
    opensfm_cameras = {}
    kapture_cameras = {cam_id: cam
                       for cam_id, cam in kapture_data.sensors.items()
                       if cam.sensor_type == 'camera'}
    for cam_id, kapture_camera in kapture_cameras.items():
        opensfm_cameras[cam_id] = export_opensfm_camera(kapture_camera)

    # export shots
    opensfm_shots = {}
    for timestamp, camera_id, image_filename in tqdm(kapture.flatten(kapture_data.records_camera),
                                                     disable=disable_tqdm):
        # retrieve pose (if there is one).
        # opensfm_shots = {image_filename: shot}
        # shot = {camera , rotation, translation, capture_time, gps_position, ...}
        opensfm_shot = {
            'capture_time': 0,  # in ms != timestamp
            'camera': camera_id,
        }
        if (timestamp, camera_id) in kapture_data.trajectories:
            pose = kapture_data.trajectories[timestamp, camera_id]
            rotation_vector = quaternion.as_rotation_vector(pose.r)
            translation_vector = pose.t.flatten()
            opensfm_shot.update({
                'rotation': rotation_vector.tolist(),
                'translation': translation_vector.tolist()
            })
        opensfm_shots[image_filename] = opensfm_shot

    # pack it
    opensfm_reconstruction = {
        'cameras': opensfm_cameras,
        'shots': opensfm_shots,
    }

    # images
    logger.info(f'writing image files "{path.join(opensfm_root_dir, "images")}".')
    image_filenames = [f for _, _, f in kapture.flatten(kapture_data.records_camera)]
    kapture_image_file_paths = [get_record_fullpath(kapture_root_dir, image_filename)
                                for image_filename in image_filenames]
    opensfm_image_file_paths = [path.join(opensfm_root_dir, 'images', image_filename)
                                for image_filename in image_filenames]
    transfer_files_from_dir(
        source_filepath_list=kapture_image_file_paths,
        destination_filepath_list=opensfm_image_file_paths,
        force_overwrite=force_overwrite_existing,
        copy_strategy=images_export_method,
    )

    _export_opensfm_features_and_matches(image_filenames, kapture_data, kapture_root_dir, opensfm_root_dir,
                                         disable_tqdm)

    # export 3D-points files
    if kapture_data.points3d is not None:
        logger.info('exporting points 3-D')
        opensfm_reconstruction['points'] = {}
        for i, (x, y, z, r, g, b) in tqdm(enumerate(kapture_data.points3d), disable=disable_tqdm):
            opensfm_reconstruction['points'][i] = {
                'coordinates': [x, y, z],
                'color': [r, g, b]
            }

    # write json files #################################################################################################
    os.makedirs(opensfm_root_dir, exist_ok=True)
    # write reconstruction.json
    opensfm_reconstruction_filepath = path.join(opensfm_root_dir, 'reconstruction.json')
    logger.info(f'writing reconstruction file "{opensfm_reconstruction_filepath}".')
    with open(opensfm_reconstruction_filepath, 'wt') as f:
        json.dump([opensfm_reconstruction], f, indent=4)

    # write camera_models.json
    opensfm_cameras_filepath = path.join(opensfm_root_dir, 'camera_models.json')
    logger.info(f'writing camera models file "{opensfm_cameras_filepath}".')
    with open(opensfm_cameras_filepath, 'wt') as f:
        json.dump(opensfm_cameras, f, indent=4)
