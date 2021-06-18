# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Openmvg constants values useful for conversion from and to.
"""

import numpy as np

from kapture.utils.Collections import AutoEnum
from enum import auto

OPENMVG_DEFAULT_JSON_FILE_NAME = 'sfm_data.json'
OPENMVG_DEFAULT_REGIONS_FILE_NAME = 'image_describer.json'
OPENMVG_SFM_DATA_VERSION_NUMBER = "0.3"

OPENMVG_DESC_HEADER_DTYPE = np.uint64
OPENMVG_DESC_HEADER_SIZE = 64  # int64 storing the number of descriptors
OPENMVG_DESC_HEADER_BYTES_NUMBER = int(OPENMVG_DESC_HEADER_SIZE / 8)  # size of a byte


# XML names
class JSON_KEY:
    SFM_DATA_VERSION = "sfm_data_version"
    ROOT_PATH = "root_path"
    VIEWS = "views"
    VIEW_PRIORS = "view_priors"
    KEY = "key"
    VALUE = "value"
    POLYMORPHIC_ID = "polymorphic_id"
    PTR_WRAPPER = "ptr_wrapper"
    ID = "id"
    DATA = "data"
    PARAMS = "params"
    LOCAL_PATH = "local_path"
    FILENAME = "filename"
    WIDTH = "width"
    HEIGHT = "height"
    ID_FEAT = "id_feat"
    ID_INTRINSIC = "id_intrinsic"
    ID_POSE = "id_pose"
    ID_VIEW = "id_view"
    INTRINSICS = "intrinsics"
    POLYMORPHIC_NAME = "polymorphic_name"
    VALUE0 = "value0"
    FOCAL_LENGTH = "focal_length"
    PRINCIPAL_POINT = "principal_point"
    DISTO_K1 = "disto_k1"
    DISTO_K3 = "disto_k3"
    DISTO_T2 = "disto_t2"
    FISHEYE = "fisheye"
    EXTRINSICS = "extrinsics"
    USE_POSE_ROTATION_PRIOR = "use_pose_rotation_prior"
    ROTATION_WEIGHT = "rotation_weight"
    ROTATION = "rotation"
    USE_POSE_CENTER_PRIOR = "use_pose_center_prior"
    CENTER_WEIGHT = "center_weight"
    CENTER = "center"
    X = "X"
    x = "x"
    STRUCTURE = "structure"
    REGIONS_TYPE = "regions_type"
    IMAGE_DESCRIBER = "image_describer"
    VALID = "valid"
    VALUE1 = "value1"
    OBSERVATIONS = "observations"
    CONTROL_POINTS = "control_points"


# Camera models
class CameraModel(AutoEnum):
    """
    Enumeration that contains all the openmvg camera model
    """
    pinhole = auto()
    pinhole_radial_k1 = auto()
    pinhole_radial_k3 = auto()
    pinhole_brown_t2 = auto()
    fisheye = auto()
