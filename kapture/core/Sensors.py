# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from kapture.utils.Collections import AutoEnum
from enum import auto
from typing import Union, Optional, List, Dict


class Sensor:
    """
    brief:  name, sensor_type, [sensor_params]+
            sensor_params is a list of string, comma separated.
    """

    def __init__(self, sensor_type: str, sensor_params: Optional[list] = None, name: Optional[str] = None):
        assert isinstance(sensor_type, str)
        assert sensor_params is None or isinstance(sensor_params, list)
        assert name is None or isinstance(name, str)
        self.name = name
        self._sensor_type = sensor_type
        self.sensor_params = sensor_params or []

    @property
    def sensor_type(self):
        """
        :return: sensor type as string
        """
        return self._sensor_type

    def __repr__(self) -> str:
        representation = ''
        representation += f'name: {self.name or "--":5} '
        representation += f'type: {self.sensor_type:6} '
        representation += '[{}]'.format(', '.join(f'{i:3}' for i in self.sensor_params))
        return representation


class Sensors(Dict[str, Sensor]):
    """
    brief: sensors[sensor_id] = <sensor>
    """

    def __setitem__(self, sensor_id: str, sensor: Sensor):
        # enforce type checking
        if not isinstance(sensor_id, str):
            raise TypeError('invalid type for sensor_id')
        if not isinstance(sensor, Sensor):
            raise TypeError('invalid type of sensor')
        super(Sensors, self).__setitem__(sensor_id, sensor)

    def __repr__(self) -> str:
        # [sensor_id] = name, type, [params]+
        representation = '\n'.join(
            f'[{sensor_id:5}] = {sensor}' for sensor_id, sensor in self.items()
        )
        return representation


class SensorType(AutoEnum):
    """
    All types of sensors
    """
    lidar = auto()
    camera = auto()
    depth = auto()
    range = auto()
    odometry = auto()
    # Usual smart-phone sensors
    accelerometer = auto()
    bluetooth = auto()
    gnss = auto()
    gyroscope = auto()
    magnetic = auto()
    wifi = auto()


class CameraType(AutoEnum):
    """
    Enumeration that contains all the supported camera types.
    Names and parameters follow COLMAP (https://github.com/colmap/colmap/tree/main/src/colmap/sensor/models),
    except OPENCV_OMNI which follows the OpenCV omnidir module.
    """
    SIMPLE_PINHOLE = auto()
    PINHOLE = auto()
    SIMPLE_RADIAL = auto()
    RADIAL = auto()
    OPENCV = auto()
    OPENCV_FISHEYE = auto()
    FULL_OPENCV = auto()
    FOV = auto()
    SIMPLE_RADIAL_FISHEYE = auto()
    RADIAL_FISHEYE = auto()
    THIN_PRISM_FISHEYE = auto()
    # added in kapture 2.0, COLMAP >= 3.12
    RAD_TAN_THIN_PRISM_FISHEYE = auto()
    SIMPLE_DIVISION = auto()
    DIVISION = auto()
    SIMPLE_FISHEYE = auto()
    FISHEYE = auto()
    EUCM = auto()
    EQUIRECTANGULAR = auto()
    SKEWED_PINHOLE = auto()
    # added in kapture 2.0, not in COLMAP
    OPENCV_OMNI = auto()
    UNKNOWN_CAMERA = auto()


CAMERA_TYPE_PARAMS_NAMES: Dict[CameraType, List[str]] = {
    # The parameter names of every camera model, in order. Image size (w, h) always comes first.
    # https://en.wikipedia.org/wiki/Pinhole_camera_model
    CameraType.SIMPLE_PINHOLE: ['w', 'h', 'f', 'cx', 'cy'],
    CameraType.PINHOLE: ['w', 'h', 'fx', 'fy', 'cx', 'cy'],
    # pinhole with a skew term s in the calibration matrix (COLMAP)
    CameraType.SKEWED_PINHOLE: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 's'],

    # Simplified versions of the OPENCV model only modeling radial distortion:
    # r^2 = u^2 + v^2; radial = k1 * r^2 + k2 * r^4; u_distorted = u + (u * radial)
    # where u,v are normalized camera coordinates. SIMPLE_RADIAL is RADIAL with k2 = 0.
    CameraType.SIMPLE_RADIAL: ['w', 'h', 'f', 'cx', 'cy', 'k'],
    CameraType.RADIAL: ['w', 'h', 'f', 'cx', 'cy', 'k1', 'k2'],

    # https://docs.opencv.org/master/d9/d0c/group__calib3d.html
    CameraType.OPENCV: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
    CameraType.FULL_OPENCV: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],

    # https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html (Kannala-Brandt)
    CameraType.OPENCV_FISHEYE: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
    # OPENCV_FISHEYE with a single focal and one or two radial coefficients
    CameraType.SIMPLE_RADIAL_FISHEYE: ['w', 'h', 'f', 'cx', 'cy', 'k'],
    CameraType.RADIAL_FISHEYE: ['w', 'h', 'f', 'cx', 'cy', 'k1', 'k2'],
    # equidistant fisheye without distortion (COLMAP)
    CameraType.SIMPLE_FISHEYE: ['w', 'h', 'f', 'cx', 'cy'],
    CameraType.FISHEYE: ['w', 'h', 'fx', 'fy', 'cx', 'cy'],

    # Frederic Devernay, Olivier Faugeras. Straight lines have to be straight: Automatic calibration and removal
    # of distortion from scenes of structured environments. Machine vision and applications, 2001.
    CameraType.FOV: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'omega'],

    # Camera Calibration with Distortion Models and Accuracy Evaluation, J Weng et al., TPAMI, 1992.
    CameraType.THIN_PRISM_FISHEYE: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1'],
    # Project Aria fisheye model (COLMAP)
    CameraType.RAD_TAN_THIN_PRISM_FISHEYE: ['w', 'h', 'fx', 'fy', 'cx', 'cy',
                                            'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'p0', 'p1', 's0', 's1', 's2', 's3'],

    # Fitzgibbon division model (COLMAP)
    CameraType.SIMPLE_DIVISION: ['w', 'h', 'f', 'cx', 'cy', 'k'],
    CameraType.DIVISION: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'k'],

    # Extended unified camera model, Khomutenko et al., 2016 (COLMAP)
    CameraType.EUCM: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 'alpha', 'beta'],

    # Unified / Mei omnidirectional model as in the OpenCV omnidir module
    # (https://docs.opencv.org/master/dd/d12/tutorial_omnidir_calib_main.html):
    # K = [fx, s, cx; 0, fy, cy; 0, 0, 1], xi the sphere shift, D = [k1, k2, p1, p2]
    CameraType.OPENCV_OMNI: ['w', 'h', 'fx', 'fy', 'cx', 'cy', 's', 'xi', 'k1', 'k2', 'p1', 'p2'],

    # 360 degrees panorama, equirectangular projection: the image size is the only parameter.
    CameraType.EQUIRECTANGULAR: ['w', 'h'],

    # Non calibrated camera
    CameraType.UNKNOWN_CAMERA: ['w', 'h'],
}

CAMERA_TYPE_PARAMS_COUNT: Dict[CameraType, int] = {
    camera_type: len(names) for camera_type, names in CAMERA_TYPE_PARAMS_NAMES.items()
}

CAMERA_TYPE_PARAMS_COUNT_FROM_NAME = {
    field.value: n
    for field, n in CAMERA_TYPE_PARAMS_COUNT.items()
}

ALL_CAMERA_SENSOR_TYPES = [SensorType.camera.name, SensorType.depth.name, SensorType.range.name]


class Camera(Sensor):
    """
    A camera definition
    """

    def __init__(self,
                 camera_type: Union[CameraType, str],
                 camera_params: list = None,
                 name: Optional[str] = None,
                 sensor_type: str = SensorType.camera.name):
        # type checking
        if name is not None and not isinstance(name, str):
            raise TypeError(f'camera name should be a string, got {type(name)}')
        if isinstance(camera_type, str):
            try:
                camera_type = CameraType[camera_type]
            except KeyError:
                raise ValueError(f'unknown camera model "{camera_type}", '
                                 f'choose among: {", ".join(t.name for t in CameraType)}') from None
        if not isinstance(camera_type, CameraType):
            raise TypeError(f'camera_type should be a CameraType or a string, got {type(camera_type)}')
        if not isinstance(camera_params, list):
            raise TypeError(f'camera_params should be a list, got {type(camera_params)}')
        expected = CAMERA_TYPE_PARAMS_NAMES[camera_type]
        if len(camera_params) != len(expected):
            raise ValueError(f'camera model {camera_type.name} expects {len(expected)} parameters '
                             f'({", ".join(expected)}), got {len(camera_params)}: {camera_params}')
        if sensor_type not in ALL_CAMERA_SENSOR_TYPES:
            raise ValueError(f'sensor type of a camera should be one of {ALL_CAMERA_SENSOR_TYPES}, got {sensor_type}')

        # make sure camera_params are numbers, store them as strings in sensor_params
        try:
            camera_params = [float(v) for v in camera_params]
        except (TypeError, ValueError):
            raise ValueError(f'camera model {camera_type.name}: parameters should be numbers, '
                             f'got {camera_params}') from None
        camera_params = [str(int(v)) if v.is_integer() else str(v) for v in camera_params]
        sensor_params = [camera_type.name] + camera_params
        super(Camera, self).__init__(sensor_type, sensor_params, name)

    @property
    def camera_type(self) -> CameraType:
        """
        :return: the camera type
        """
        if self.sensor_params is None:
            raise ValueError('sensor_params should not be None for a camera')
        return CameraType[self.sensor_params[0]]

    @property
    def camera_params(self) -> List[float]:
        """
        :return: camera parameters
        """
        if self.sensor_params is None:
            raise ValueError('sensor_params should not be None for a camera')
        num_params = CAMERA_TYPE_PARAMS_COUNT[self.camera_type]
        return [float(c) for c in self.sensor_params[1:(1+num_params)]]

    @property
    def camera_params_names(self) -> List[str]:
        """
        :return: names of the camera parameters, in the same order as camera_params (eg. ['w', 'h', 'f', 'cx', 'cy'])
        """
        return CAMERA_TYPE_PARAMS_NAMES[self.camera_type]

    def camera_param(self, name: str) -> float:
        """
        Get a camera parameter by its name, eg. camera.camera_param('fx').

        :param name: parameter name, see CAMERA_TYPE_PARAMS_NAMES
        :return: the parameter value
        """
        names = self.camera_params_names
        if name not in names:
            raise KeyError(f'camera model {self.camera_type.name} has no parameter "{name}" (has: {", ".join(names)})')
        return self.camera_params[names.index(name)]


def create_sensor(sensor_type: str, sensor_params: Optional[list] = None, name: Optional[str] = None):
    """
    Creates an instance of a sensor

    :param sensor_type: type of sensor ('camera', ...)
    :param sensor_params: sensor specific parameters
    :param name: sensor name
    :return: created instance
    """
    if sensor_type in ALL_CAMERA_SENSOR_TYPES:
        if not sensor_params:
            raise ValueError(f'a {sensor_type} sensor needs parameters: camera model, then model parameters')
        return Camera(camera_type=sensor_params[0], camera_params=sensor_params[1:], name=name, sensor_type=sensor_type)
    else:
        return Sensor(sensor_type, sensor_params, name)
