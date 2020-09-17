# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
All kapture objects representing the data kapture manages.
"""

from .PoseTransform import PoseTransform  # noqa: F401
from .Sensors import Sensor, Sensors, create_sensor  # noqa: F401
from .Sensors import Camera, CameraType, CAMERA_TYPE_PARAMS_COUNT, CAMERA_TYPE_PARAMS_COUNT_FROM_NAME  # noqa: F401
from .Rigs import Rigs  # noqa: F401
from .Trajectories import Trajectories  # noqa: F401
from .Trajectories import rigs_remove, rigs_remove_inplace, rigs_recover, rigs_recover_inplace  # noqa: F401
from .Records import RecordsBase, RecordsArray, RecordsCamera, RecordsDepth, RecordsLidar  # noqa: F401
from .Records import RecordsWifi, RecordWifi, RecordWifiSignal  # noqa: F401
from .Records import RecordBluetooth, RecordsBluetooth, RecordBluetoothSignal  # noqa: F401
from .Records import RecordGnss, RecordsGnss  # noqa: F401
from .Records import RecordAccelerometer, RecordsAccelerometer  # noqa: F401
from .Records import RecordGyroscope, RecordsGyroscope  # noqa: F401
from .Records import RecordMagnetic, RecordsMagnetic  # noqa: F401
from .ImageFeatures import Keypoints, Descriptors, GlobalFeatures  # noqa: F401
from .Observations import Observations  # noqa: F401
from .Matches import Matches  # noqa: F401
from .Points3d import Points3d  # noqa: F401
from .Kapture import Kapture  # noqa: F401
from .flatten import flatten  # noqa: F401

import logging

# it is up to the library client to specify the log handler.
logging.getLogger("kapture").addHandler(logging.NullHandler())
