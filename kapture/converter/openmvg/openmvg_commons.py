# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Openmvg constants values useful for conversion from and to.
"""

from kapture.utils.Collections import AutoEnum
from enum import auto

DEFAULT_JSON_FILE_NAME = 'sfm_data.json'
SFM_DATA_VERSION_NUMBER = "0.3"

# XML names
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
LOCAL_PATH = "local_path"
FILENAME = "filename"
WIDTH = "width"
HEIGHT = "height"
ID_VIEW = "id_view"
ID_INTRINSIC = "id_intrinsic"
ID_POSE = "id_pose"
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
STRUCTURE = "structure"
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
