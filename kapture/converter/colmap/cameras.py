# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Colmap camera management
"""

import kapture


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
]

CAMERA_MODEL_NAMES = {i: name for name, i in CAMERA_MODEL_NAME_ID}
CAMERA_MODEL_IDS = {name: i for name, i in CAMERA_MODEL_NAME_ID}

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


def get_colmap_camera(camera: kapture.Camera):
    """
    Compute the colmap camera definition.

    :param camera: a kapture camera definition
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
    elif camera.camera_type.value in CAMERA_MODEL_IDS:
        prior_focal_length = True
        camera_id = CAMERA_MODEL_IDS[camera.camera_type.value]
        params = camera.camera_params[2:]
    else:
        raise ValueError(
            'This sensor model: {} is not supported by colmap'.format(camera.camera_type.value))
    return camera_id, width, height, params, prior_focal_length
