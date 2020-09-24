# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from .Sensors import Sensors, Camera
from .Rigs import Rigs
from .Trajectories import Trajectories
from .Records import RecordsCamera, RecordsDepth, RecordsLidar, RecordsWifi, RecordsBluetooth
from .Records import RecordsGnss, RecordsAccelerometer, RecordsGyroscope, RecordsMagnetic
from .ImageFeatures import Keypoints, Descriptors, GlobalFeatures
from .Matches import Matches
from .Observations import Observations
from .Points3d import Points3d
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Kapture:
    """
    brief: Root class of all kapture data.
    """

    def __init__(
            self,
            sensors: Optional[Sensors] = None,
            rigs: Optional[Rigs] = None,
            trajectories: Optional[Trajectories] = None,
            records_camera: Optional[RecordsCamera] = None,
            records_depth: Optional[RecordsDepth] = None,
            records_lidar: Optional[RecordsLidar] = None,
            records_wifi: Optional[RecordsWifi] = None,
            records_bluetooth: Optional[RecordsBluetooth] = None,
            records_gnss: Optional[RecordsGnss] = None,
            records_accelerometer: Optional[RecordsAccelerometer] = None,
            records_gyroscope: Optional[RecordsGyroscope] = None,
            records_magnetic: Optional[RecordsMagnetic] = None,
            keypoints: Optional[Keypoints] = None,
            descriptors: Optional[Descriptors] = None,
            global_features: Optional[GlobalFeatures] = None,
            matches: Optional[Matches] = None,
            observations: Optional[Observations] = None,
            points3d: Optional[Points3d] = None,
    ):
        self.sensors = sensors
        self.rigs = rigs
        self.trajectories = trajectories
        self.records_camera = records_camera
        self.records_depth = records_depth
        self.records_lidar = records_lidar
        self.records_wifi = records_wifi
        self.records_bluetooth = records_bluetooth
        self.records_gnss = records_gnss
        self.records_accelerometer = records_accelerometer
        self.records_gyroscope = records_gyroscope
        self.records_magnetic = records_magnetic
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.global_features = global_features
        self.matches = matches
        self.observations = observations
        self.points3d = points3d
        self.__version__ = '0.0'

    @property
    def format_version(self):
        """
        :return: kapture format version
        """
        return self.__version__

    @property
    def sensors(self) -> Optional[Sensors]:
        """
        :return: the list of sensors
        """
        return self._sensors

    @sensors.setter
    def sensors(self, sensors: Optional[Sensors]):
        if sensors is not None and not isinstance(sensors, Sensors):
            raise TypeError('sensors expected')
        self._sensors = sensors

    @property
    def cameras(self) -> Dict[str, Camera]:
        """
        :return: the cameras (particular kind of sensor) as dictionary keyed by the camera identifier
        """
        if self.sensors is None:
            return {}
        return {k: v for k, v in self.sensors.items() if isinstance(v, Camera)}

    @property
    def rigs(self) -> Optional[Rigs]:
        """
        :return: the list of rigs
        """
        return self._rigs

    @rigs.setter
    def rigs(self, rigs: Optional[Rigs]):
        if rigs is not None and not isinstance(rigs, Rigs):
            raise TypeError('rigs expected')
        self._rigs = rigs

    @property
    def trajectories(self) -> Optional[Trajectories]:
        """
        :return: the list of trajectories
        """
        return self._trajectories

    @trajectories.setter
    def trajectories(self, trajectories: Optional[Trajectories]):
        if trajectories is not None and not isinstance(trajectories, Trajectories):
            raise TypeError('trajectories expected')
        self._trajectories = trajectories

    @property
    def records_camera(self) -> Optional[RecordsCamera]:
        """
        :return: Camera records
        """
        return self._records_camera

    @records_camera.setter
    def records_camera(self, records_camera: Optional[RecordsCamera]):
        if records_camera is not None and not isinstance(records_camera, RecordsCamera):
            raise TypeError('RecordsCamera expected')
        self._records_camera = records_camera

    @property
    def records_depth(self) -> Optional[RecordsDepth]:
        """
        :return: Depth records
        """
        return self._records_depth

    @records_depth.setter
    def records_depth(self, records_depth: Optional[RecordsDepth]):
        if records_depth is not None and not isinstance(records_depth, RecordsDepth):
            raise TypeError('RecordsDepth expected')
        self._records_depth = records_depth

    @property
    def records_lidar(self) -> Optional[RecordsLidar]:
        """
        :return: Lidar records
        """
        return self._records_lidar

    @records_lidar.setter
    def records_lidar(self, records_lidar: Optional[RecordsLidar]):
        if records_lidar is not None and not isinstance(records_lidar, RecordsLidar):
            raise TypeError('RecordsLidar expected')
        self._records_lidar = records_lidar

    @property
    def records_wifi(self) -> Optional[RecordsWifi]:
        """
        :return: Wifi records
        """
        return self._records_wifi

    @records_wifi.setter
    def records_wifi(self, records_wifi: Optional[RecordsWifi]):
        if records_wifi is not None and not isinstance(records_wifi, RecordsWifi):
            raise TypeError('RecordsWifi expected')
        self._records_wifi = records_wifi

    @property
    def records_bluetooth(self) -> Optional[RecordsBluetooth]:
        """
        :return: Bluetooth records
        """
        return self._records_bluetooth

    @records_bluetooth.setter
    def records_bluetooth(self, records_bluetooth: Optional[RecordsBluetooth]):
        if records_bluetooth is not None and not isinstance(records_bluetooth, RecordsBluetooth):
            raise TypeError('RecordsBluetooth expected')
        self._records_bluetooth = records_bluetooth

    @property
    def records_gnss(self) -> Optional[RecordsGnss]:
        """
        :return: GNSS records
        """
        return self._records_gnss

    @records_gnss.setter
    def records_gnss(self, records_gnss: Optional[RecordsGnss]):
        if records_gnss is not None and not isinstance(records_gnss, RecordsGnss):
            raise TypeError('RecordsGnss expected')
        self._records_gnss = records_gnss

    @property
    def records_accelerometer(self) -> Optional[RecordsAccelerometer]:
        """
        :return: Acceleration records
        """
        return self._records_accelerometer

    @records_accelerometer.setter
    def records_accelerometer(self, records_accelerometer: Optional[RecordsAccelerometer]):
        if records_accelerometer is not None and not isinstance(records_accelerometer, RecordsAccelerometer):
            raise TypeError('RecordsAccelerometer expected')
        self._records_accelerometer = records_accelerometer

    @property
    def records_gyroscope(self) -> Optional[RecordsGyroscope]:
        """
        :return: Gyroscope records
        """
        return self._records_gyroscope

    @records_gyroscope.setter
    def records_gyroscope(self, records_gyroscope: Optional[RecordsGyroscope]):
        if records_gyroscope is not None and not isinstance(records_gyroscope, RecordsGyroscope):
            raise TypeError('RecordsGyroscope expected')
        self._records_gyroscope = records_gyroscope

    @property
    def records_magnetic(self) -> Optional[RecordsMagnetic]:
        """
        :return: Magnetic records
        """
        return self._records_magnetic

    @records_magnetic.setter
    def records_magnetic(self, records_magnetic: Optional[RecordsMagnetic]):
        if records_magnetic is not None and not isinstance(records_magnetic, RecordsMagnetic):
            raise TypeError('RecordsMagnetic expected')
        self._records_magnetic = records_magnetic

    @property
    def keypoints(self) -> Optional[Keypoints]:
        """
        :return: the keypoints
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(self, keypoints: Optional[Keypoints]):
        if keypoints is not None and not isinstance(keypoints, Keypoints):
            raise TypeError('Keypoints expected')
        self._keypoints = keypoints

    @property
    def descriptors(self) -> Optional[Descriptors]:
        """
        :return: the descriptors
        """
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors: Optional[Descriptors]):
        if descriptors is not None and not isinstance(descriptors, Descriptors):
            raise TypeError('Descriptors expected')
        self._descriptors = descriptors

    @property
    def global_features(self) -> Optional[GlobalFeatures]:
        """
        :return: the global features
        """
        return self._global_features

    @global_features.setter
    def global_features(self, global_features: Optional[GlobalFeatures]):
        if global_features is not None and not isinstance(global_features, GlobalFeatures):
            raise TypeError('GlobalFeatures expected')
        self._global_features = global_features

    @property
    def matches(self) -> Optional[Matches]:
        """
        :return: the matches
        """
        return self._matches

    @matches.setter
    def matches(self, matches: Optional[Matches]):
        if matches is not None and not isinstance(matches, Matches):
            raise TypeError('Matches expected')
        self._matches = matches

    @property
    def observations(self) -> Optional[Observations]:
        """
        :return: the observations
        """
        return self._observations

    @observations.setter
    def observations(self, observations: Optional[Observations]):
        if observations is not None and not isinstance(observations, Observations):
            raise TypeError('Observations expected')
        self._observations = observations

    @property
    def points3d(self) -> Optional[Points3d]:
        """
        :return: the 3D points
        """
        return self._points3d

    @points3d.setter
    def points3d(self, points3d: Optional[Points3d]):
        if points3d is not None and not isinstance(points3d, Points3d):
            raise TypeError('Points3d expected')
        self._points3d = points3d

    def as_dict(self, keep_none=False) -> Dict[str, Union[
        Sensors,
        Rigs,
        Trajectories,
        RecordsCamera,
        RecordsDepth,
        RecordsLidar,
        RecordsWifi,
        RecordsBluetooth,
        RecordsGnss,
        RecordsAccelerometer,
        RecordsGyroscope,
        RecordsMagnetic,
        Keypoints,
        Descriptors,
        GlobalFeatures,
        Matches,
        Observations,
        Points3d
    ]]:
        """ convenience accessor to all members at once """
        return {
            name[1:]: member
            for name, member in self.__dict__.items()
            if not name.startswith('__') and (keep_none or member is not None)
        }

    def __repr__(self) -> str:
        representation = '\n'.join([
            f'{name:14} : {len(value):4}'
            for name, value in self.as_dict().items()
        ])

        if len(representation) == 0:
            representation = 'no data'
        return representation
