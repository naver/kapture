# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Colmap camera management
"""

import logging
import kapture

logger = logging.getLogger('colmap')

# see https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models/base.h
CAMERA_MODEL_NAME_ID = [
    ('SIMPLE_PINHOLE', 0),
    ('PINHOLE', 1),
    ('SIMPLE_RADIAL', 2),
    ('RADIAL', 3),
    ('OPENCV', 4),
    ('OPENCV_FISHEYE', 5),
    ('FULL_OPENCV', 6),
    ('FOV', 7),
    ('SIMPLE_RADIAL_FISHEYE', 8),
    ('RADIAL_FISHEYE', 9),
    ('THIN_PRISM_FISHEYE', 10),
    # the following need COLMAP >= 3.12
    ('RAD_TAN_THIN_PRISM_FISHEYE', 11),
    ('SIMPLE_DIVISION', 12),
    ('DIVISION', 13),
    ('SIMPLE_FISHEYE', 14),
    ('FISHEYE', 15),
    ('EUCM', 16),
    ('EQUIRECTANGULAR', 17),
    ('SKEWED_PINHOLE', 18),
]
FIRST_COLMAP_3_12_MODEL_ID = 11

CAMERA_MODEL_NAMES = {i: name for name, i in CAMERA_MODEL_NAME_ID}
CAMERA_MODEL_IDS = {name: i for name, i in CAMERA_MODEL_NAME_ID}
# kapture camera models that have no COLMAP equivalent
COLMAP_UNSUPPORTED_CAMERA_TYPES = []

DEFAULT_FOCAL_LENGTH_FACTOR = 1.2


def get_camera_kapture_id_from_colmap_id(camera_id_colmap) -> str:
    """
    Create a deterministic kapture camera identifier from the colmap camera identifier:
    sensor_id = "cam_xxxxx"  where "xxxxx" is the colmap ID.

    :param camera_id_colmap: colmap camera identifier
    :return: kapture camera identifier.
    """
    assert isinstance(camera_id_colmap, int)
    return f'cam_{camera_id_colmap:05d}'


def omni_to_eucm_params(camera: kapture.Camera):
    """
    Converts the unified (Mei) model parameters of an OPENCV_OMNI camera to the COLMAP EUCM parameters.

    The unified model with mirror parameter xi is the extended unified model with beta = 1 and
    alpha = xi / (1 + xi), the focal lengths being scaled by 1 / (1 + xi).
    Skew and distortion coefficients (k1, k2, p1, p2) have no equivalent in EUCM and are dropped.

    :param camera: an OPENCV_OMNI kapture camera
    :return: EUCM parameters [fx, fy, cx, cy, alpha, beta]
    """
    assert camera.camera_type == kapture.CameraType.OPENCV_OMNI
    xi = camera.camera_param('xi')
    scale = 1.0 / (1.0 + xi)
    return [camera.camera_param('fx') * scale, camera.camera_param('fy') * scale,
            camera.camera_param('cx'), camera.camera_param('cy'),
            xi * scale, 1.0]


def get_colmap_camera(camera: kapture.Camera):
    """
    Compute the colmap camera definition.

    :param camera: a kapture camera definition
    OPENCV_OMNI is approximated by EUCM (see omni_to_eucm_params).

    :return: colmap camera parameters.
    """
    assert isinstance(camera, kapture.Camera)
    assert len(camera.camera_params) >= 2

    width = camera.camera_params[0]
    height = camera.camera_params[1]
    if camera.camera_type == kapture.CameraType.UNKNOWN_CAMERA:
        prior_focal_length = False
        camera_id = CAMERA_MODEL_IDS[kapture.CameraType.SIMPLE_RADIAL.value]
        focal_length = DEFAULT_FOCAL_LENGTH_FACTOR * max(width, height)
        cx = width/2.0
        cy = height/2.0
        params = [focal_length, cx, cy, 0.0]
    elif camera.camera_type == kapture.CameraType.OPENCV_OMNI:
        prior_focal_length = True
        camera_id = CAMERA_MODEL_IDS['EUCM']
        params = omni_to_eucm_params(camera)
        logger.warning('camera model OPENCV_OMNI exported as EUCM (requires COLMAP >= 3.12): '
                       'skew and distortion (k1, k2, p1, p2) are dropped')
    elif camera.camera_type.value in CAMERA_MODEL_IDS:
        prior_focal_length = True
        camera_id = CAMERA_MODEL_IDS[camera.camera_type.value]
        params = camera.camera_params[2:]
        if camera_id >= FIRST_COLMAP_3_12_MODEL_ID:
            logger.warning(f'camera model {camera.camera_type.value} requires COLMAP >= 3.12')
    else:
        raise ValueError(f'camera model {camera.camera_type.value} cannot be exported to COLMAP')
    return camera_id, width, height, params, prior_focal_length


def get_kapture_camera(model_name_or_id, width, height, params, colmap_camera_id=None) -> kapture.Camera:
    """
    Compute the kapture camera from a colmap camera definition.
    A camera model unknown to kapture is converted to UNKNOWN_CAMERA (with a warning).

    :param model_name_or_id: colmap camera model, by name (cameras.txt) or by id (database)
    :param width: image width
    :param height: image height
    :param params: colmap camera parameters (without image size)
    :param colmap_camera_id: colmap camera identifier, for the warning
    :return: kapture camera
    """
    if isinstance(model_name_or_id, int):
        model_name = CAMERA_MODEL_NAMES.get(model_name_or_id, str(model_name_or_id))
    else:
        model_name = str(model_name_or_id)
    if model_name not in kapture.CAMERA_TYPE_PARAMS_COUNT_FROM_NAME:
        logger.warning(f'unknown colmap camera model ({model_name}) for camera {colmap_camera_id}: '
                       f'using {kapture.CameraType.UNKNOWN_CAMERA.name}')
        return kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, [width, height])
    return kapture.Camera(model_name, [width, height] + list(params))
