# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
All kapture objects representing the data kapture manages.
"""

from .PoseTransform import PoseTransform
from .Sensors import Sensor, Sensors, create_sensor
from .Sensors import Camera, CameraType, CAMERA_TYPE_PARAMS_COUNT, CAMERA_TYPE_PARAMS_COUNT_FROM_NAME
from .Rigs import Rigs
from .Trajectories import Trajectories, rigs_remove, rigs_remove_inplace, rigs_recover, rigs_recover_inplace
from .Records import RecordsBase, RecordsCamera, RecordsLidar, RecordsWifi, RecordWifi, RecordGnss, RecordsGnss
from .ImageFeatures import Keypoints, Descriptors, GlobalFeatures
from .Observations import Observations
from .Matches import Matches
from .Points3d import Points3d
from .Kapture import Kapture
from .flatten import flatten

import logging

# it is up to the library client to specify the log handler.
logging.getLogger("kapture").addHandler(logging.NullHandler())
