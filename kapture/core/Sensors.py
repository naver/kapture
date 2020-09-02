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


class CameraType(AutoEnum):
    """
    Enumeration that contains all the supported camera types
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
    UNKNOWN_CAMERA = auto()


CAMERA_TYPE_PARAMS_COUNT = {
    # https://en.wikipedia.org/wiki/Pinhole_camera_model
    # w, h, f, cx, cy
    CameraType.SIMPLE_PINHOLE: 5,

    # https://en.wikipedia.org/wiki/Pinhole_camera_model
    # w, h, fx, fy, cx, cy
    CameraType.PINHOLE: 6,

    # Simplified versions of the OPENCV model only modeling radial distortion.
    # w, h, f, cx, cy, k
    # equivalent to CameraType.RADIAL with k1=k, k2=0
    # see: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h#L715
    CameraType.SIMPLE_RADIAL: 6,

    # Simplified versions of the OPENCV model only modeling radial distortion
    # w, h, f, cx, cy, k1, k2
    # r^2 = u^2 + v^2;
    # radial = k1 * r^2 + k2 * r^2 * r^2
    # u_distorted = u + (u * radial)
    # where u,v are normalized camera coordinates
    # see: https://github.com/colmap/colmap/blob/master/src/base/camera_models.h#L784
    CameraType.RADIAL: 7,

    # http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # w, h, fx, fy, cx, cy, k1, k2, p1, p2
    CameraType.OPENCV: 10,

    # http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # w, h, fx, fy, cx, cy, k1, k2, k3, k4
    CameraType.OPENCV_FISHEYE: 10,

    # http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    CameraType.FULL_OPENCV: 14,

    # Frederic Devernay, Olivier Faugeras. Straight lines have to be straight:
    # Automatic calibration and removal of distortion from scenes of structured
    # environments. Machine vision and applications, 2001.
    # w, h, fx, fy, cx, cy, omega
    CameraType.FOV: 7,

    # equivalent to the OpenCVFisheyeCameraModel but has only one
    # radial distortion coefficient.
    # w, h, f, cx, cy, k
    CameraType.SIMPLE_RADIAL_FISHEYE: 6,

    # equivalent to the OpenCVFisheyeCameraModel but has only two
    # radial distortion coefficients.
    # w, h, f, cx, cy, k1, k2
    CameraType.RADIAL_FISHEYE: 7,

    # Camera Calibration with Distortion Models and Accuracy Evaluation,
    # J Weng et al., TPAMI, 1992.
    # w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sx2
    CameraType.THIN_PRISM_FISHEYE: 14,

    # Non calibrated camera
    # w, h
    CameraType.UNKNOWN_CAMERA: 2
}

CAMERA_TYPE_PARAMS_COUNT_FROM_NAME = {
    field.value: n
    for field, n in CAMERA_TYPE_PARAMS_COUNT.items()
}


class Camera(Sensor):
    """
    A camera definition
    """

    def __init__(self,
                 camera_type: Union[CameraType, str],
                 camera_params: list = None,
                 name: Optional[str] = None,
                 sensor_type: str = 'camera'):
        # type checking
        assert name is None or isinstance(name, str)
        if isinstance(camera_type, str):
            camera_type = CameraType[camera_type]
        assert isinstance(camera_type, CameraType)
        # check params are consistent with model
        assert isinstance(camera_params, list)
        assert len(camera_params) == CAMERA_TYPE_PARAMS_COUNT[camera_type]
        assert sensor_type == 'camera' or sensor_type == 'depth'

        # make sure it crashes if camera_params cannot be cast to float, store as string in sensor_params
        camera_params = [float(v) for v in camera_params]
        camera_params = [str(int(v)) if v.is_integer() else str(v) for v in camera_params]
        sensor_params = [camera_type.name] + camera_params
        super(Camera, self).__init__(sensor_type=sensor_type, sensor_params=sensor_params, name=name)

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


def create_sensor(sensor_type: str, sensor_params: Optional[list] = None, name: Optional[str] = None):
    """
    Creates a instance of a sensor

    :param sensor_type: type of sensor ('camera', ...)
    :param sensor_params: sensor specific parameters
    :param name: sensor name
    :return: created instance
    """
    if sensor_type == 'camera' or sensor_type == 'depth':
        assert sensor_params is not None
        return Camera(camera_type=sensor_params[0], camera_params=sensor_params[1:], name=name, sensor_type=sensor_type)
    else:
        return Sensor(sensor_type=sensor_type, sensor_params=sensor_params, name=name)
