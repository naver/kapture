# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Export kapture lidar, IMU (and optionally camera) records to a ROS2 bag, for off-the-shelf SLAMs (GLIM, FAST-LIO2, ...).

The bag is written in pure python with the `rosbags` package (pip install rosbags), no ROS install needed.

Topics:
 - /<lidar_id>/points: sensor_msgs/PointCloud2 per lidar record, fields x y z (float32), intensity (float32) and
   t (uint32, ns since the scan start). The header stamp is the scan start, the kapture record timestamp being the
   time of the last point. Points are kept in the lidar frame of the point cloud file.
 - /<imu_id>/imu: sensor_msgs/Imu, from the kapture sensors <imu_id>_accel (accelerometer, m/s2) and <imu_id>_gyro
   (gyroscope, rad/s), one message per accelerometer record (gyroscope interpolated if not recorded at the same time).
 - /tf_static: tf2_msgs/TFMessage, one transform per rig entry (parent: rig id, child: sensor id).
 - /<camera_id>/image/compressed: sensor_msgs/CompressedImage (jpeg / png bytes copied as is), and
   /<camera_id>/camera_info for the pinhole camera models.
"""

import heapq
import logging
import os.path as path
import shutil
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import kapture
import kapture.io.csv as csv
from kapture.io.ply import read_ply
from kapture.io.records import get_record_fullpath

logger = logging.getLogger('export_ros2bag')

try:
    from rosbags.rosbag2 import Writer, StoragePlugin, CompressionFormat, CompressionMode
    from rosbags.typesys import Stores, get_typestore
    has_rosbags = True
except ImportError:  # postpone the error to the actual use.
    has_rosbags = False

ACCEL_SUFFIX = '_accel'
GYRO_SUFFIX = '_gyro'
# kapture camera models convertible to a ROS CameraInfo (plumb_bob distortion: k1 k2 p1 p2 k3)
PINHOLE_CAMERA_INFO_MODELS = {
    kapture.CameraType.SIMPLE_PINHOLE.name,  # w h f cx cy
    kapture.CameraType.PINHOLE.name,  # w h fx fy cx cy
    kapture.CameraType.SIMPLE_RADIAL.name,  # w h f cx cy k1
    kapture.CameraType.RADIAL.name,  # w h f cx cy k1 k2
    kapture.CameraType.OPENCV.name,  # w h fx fy cx cy k1 k2 p1 p2
}

Message = Tuple[int, object, bytes]  # (log time ns, rosbags connection, serialized message)


def us_to_ns(timestamp_us: int) -> int:
    return int(timestamp_us) * 1000


def pose_to_tum(pose: kapture.PoseTransform) -> List[float]:
    """ :return: [x, y, z, qx, qy, qz, qw] of the pose. """
    r, t = pose.r, pose.t.ravel()
    return [float(t[0]), float(t[1]), float(t[2]), float(r.x), float(r.y), float(r.z), float(r.w)]


def find_imu_ids(sensors: kapture.Sensors) -> List[str]:
    """ IMU ids = names such that <name>_accel is an accelerometer and <name>_gyro a gyroscope sensor. """
    imu_ids = []
    for sensor_id, sensor in sensors.items():
        if sensor.sensor_type == kapture.SensorType.accelerometer.name and sensor_id.endswith(ACCEL_SUFFIX):
            imu_id = sensor_id[:-len(ACCEL_SUFFIX)]
            gyro = sensors.get(imu_id + GYRO_SUFFIX)
            if gyro is not None and gyro.sensor_type == kapture.SensorType.gyroscope.name:
                imu_ids.append(imu_id)
    return sorted(imu_ids)


def lidar_imu_extrinsics(rigs: kapture.Rigs, lidar_id: str, imu_sensor_id: str) -> Optional[kapture.PoseTransform]:
    """ :return: T_lidar_imu, the pose of the IMU in the lidar frame (imu points to lidar points), or None. """
    for rig_id in rigs:
        if lidar_id in rigs[rig_id] and imu_sensor_id in rigs[rig_id]:
            # rigs hold rig -> sensor transforms: T_lidar_imu = T_lidar_rig o T_rig_imu
            return kapture.PoseTransform.compose([rigs[rig_id, lidar_id], rigs[rig_id, imu_sensor_id].inverse()])
    return None


class Ros2BagExporter:
    """ Builds and serializes the ROS2 messages with a rosbags typestore. """

    def __init__(self, typestore=None):
        if not has_rosbags:
            raise ImportError('exporting to ROS2 bags requires the python package rosbags (pip install rosbags)')
        self.typestore = typestore or get_typestore(Stores.ROS2_HUMBLE)
        types = self.typestore.types
        self.Header = types['std_msgs/msg/Header']
        self.Time = types['builtin_interfaces/msg/Time']
        self.PointCloud2 = types['sensor_msgs/msg/PointCloud2']
        self.PointField = types['sensor_msgs/msg/PointField']
        self.Imu = types['sensor_msgs/msg/Imu']
        self.Vector3 = types['geometry_msgs/msg/Vector3']
        self.Quaternion = types['geometry_msgs/msg/Quaternion']
        self.Transform = types['geometry_msgs/msg/Transform']
        self.TransformStamped = types['geometry_msgs/msg/TransformStamped']
        self.TFMessage = types['tf2_msgs/msg/TFMessage']
        self.CompressedImage = types['sensor_msgs/msg/CompressedImage']
        self.CameraInfo = types['sensor_msgs/msg/CameraInfo']
        self.RegionOfInterest = types['sensor_msgs/msg/RegionOfInterest']
        self._point_field_types = {'f4': self.PointField.FLOAT32, 'f8': self.PointField.FLOAT64,
                                   'u1': self.PointField.UINT8, 'u2': self.PointField.UINT16,
                                   'u4': self.PointField.UINT32, 'i1': self.PointField.INT8,
                                   'i2': self.PointField.INT16, 'i4': self.PointField.INT32}

    def header(self, timestamp_ns: int, frame_id: str):
        return self.Header(stamp=self.Time(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000),
                           frame_id=frame_id)

    def serialize(self, message) -> bytes:
        return self.typestore.serialize_cdr(message, message.__msgtype__)

    def point_cloud2(self, timestamp_ns: int, frame_id: str, points: np.ndarray):
        """
        :param points: structured array, one field per point cloud field (little endian scalar types).
        """
        fields = []
        for name in points.dtype.names:
            dtype = points.dtype.fields[name][0]
            offset = points.dtype.fields[name][1]
            fields.append(self.PointField(name=name, offset=offset,
                                          datatype=self._point_field_types[dtype.kind + str(dtype.itemsize)],
                                          count=1))
        return self.PointCloud2(header=self.header(timestamp_ns, frame_id), height=1, width=len(points),
                                fields=fields, is_bigendian=False, point_step=points.dtype.itemsize,
                                row_step=points.dtype.itemsize * len(points),
                                data=np.ascontiguousarray(points).view(np.uint8), is_dense=True)

    def imu(self, timestamp_ns: int, frame_id: str, accel: np.ndarray, gyro: np.ndarray):
        unknown = np.zeros(9)
        unknown[0] = -1  # ROS convention: orientation not available
        return self.Imu(header=self.header(timestamp_ns, frame_id),
                        orientation=self.Quaternion(x=0., y=0., z=0., w=1.),
                        orientation_covariance=unknown,
                        angular_velocity=self.Vector3(x=float(gyro[0]), y=float(gyro[1]), z=float(gyro[2])),
                        angular_velocity_covariance=np.zeros(9),
                        linear_acceleration=self.Vector3(x=float(accel[0]), y=float(accel[1]), z=float(accel[2])),
                        linear_acceleration_covariance=np.zeros(9))

    def tf_message(self, timestamp_ns: int, rigs: kapture.Rigs):
        transforms = []
        for rig_id, sensor_id, pose_sensor_from_rig in kapture.flatten(rigs, is_sorted=True):
            x, y, z, qx, qy, qz, qw = pose_to_tum(pose_sensor_from_rig.inverse())  # tf: child pose in the parent
            transforms.append(self.TransformStamped(
                header=self.header(timestamp_ns, rig_id), child_frame_id=sensor_id,
                transform=self.Transform(translation=self.Vector3(x=x, y=y, z=z),
                                         rotation=self.Quaternion(x=qx, y=qy, z=qz, w=qw))))
        return self.TFMessage(transforms=transforms)

    def compressed_image(self, timestamp_ns: int, frame_id: str, image_format: str, data: bytes):
        return self.CompressedImage(header=self.header(timestamp_ns, frame_id), format=image_format,
                                    data=np.frombuffer(data, dtype=np.uint8))

    def camera_info(self, timestamp_ns: int, frame_id: str, camera: kapture.Camera):
        """ :return: a CameraInfo for the pinhole models of PINHOLE_CAMERA_INFO_MODELS, None otherwise. """
        if camera.camera_type.name not in PINHOLE_CAMERA_INFO_MODELS:
            return None
        params = [float(p) for p in camera.camera_params]
        width, height = int(params[0]), int(params[1])
        if camera.camera_type in (kapture.CameraType.PINHOLE, kapture.CameraType.OPENCV):
            fx, fy, cx, cy = params[2:6]
            distortion = params[6:]
        else:
            fx = fy = params[2]
            cx, cy = params[3:5]
            distortion = params[5:]
        distortion = (distortion + [0.] * 5)[:5]  # plumb_bob: k1 k2 p1 p2 k3
        k = np.array([fx, 0., cx, 0., fy, cy, 0., 0., 1.])
        p = np.array([fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.])
        return self.CameraInfo(header=self.header(timestamp_ns, frame_id), height=height, width=width,
                               distortion_model='plumb_bob', d=np.array(distortion), k=k, r=np.eye(3).ravel(), p=p,
                               binning_x=0, binning_y=0,
                               roi=self.RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))


def lidar_points_to_cloud(vertices: np.ndarray, time_field: Optional[str], intensity_field: Optional[str]
                          ) -> Tuple[np.ndarray, int]:
    """
    Converts PLY vertices to point cloud fields.

    :param vertices: structured array with x, y, z and optionally the time and intensity fields.
    :param time_field: name of the per-point time field (ns, any origin), or None.
    :param intensity_field: name of the intensity field, or None.
    :return: structured array (x, y, z, [intensity], [t]) and the scan duration (ns, last point - first point).
    """
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if intensity_field:
        fields.append(('intensity', '<f4'))
    if time_field:
        fields.append(('t', '<u4'))
    points = np.empty(len(vertices), dtype=np.dtype(fields))
    points['x'], points['y'], points['z'] = vertices['x'], vertices['y'], vertices['z']
    if intensity_field:
        points['intensity'] = vertices[intensity_field]
    duration_ns = 0
    if time_field and len(vertices):
        times = np.asarray(vertices[time_field], dtype=np.float64)
        t0 = times.min()
        duration_ns = int(round(times.max() - t0))
        points['t'] = np.round(times - t0).astype(np.uint32)
    return points, duration_ns


def kapture_to_ros2bag(kapture_dirpath: str,  # noqa: C901
                       bag_dirpath: str,
                       lidar_ids: Optional[List[str]] = None,
                       imu_ids: Optional[List[str]] = None,
                       camera_ids: Optional[List[str]] = None,
                       time_field: Optional[str] = 'timestamps_0',
                       intensity_field: Optional[str] = 'intensities_0',
                       storage: str = 'sqlite3',
                       compression: str = 'none',
                       force_overwrite_existing: bool = False) -> Dict[str, int]:
    """
    Exports kapture lidar, IMU and camera records to a ROS2 bag.

    :param kapture_dirpath: input kapture directory.
    :param bag_dirpath: output ROS2 bag directory (created, must not exist unless force_overwrite_existing).
    :param lidar_ids: lidars to export (default: all lidars with records). Topic /<lidar_id>/points.
    :param imu_ids: IMUs to export, as <imu_id> for the kapture sensors <imu_id>_accel + <imu_id>_gyro
                    (default: all such pairs). Topic /<imu_id>/imu.
    :param camera_ids: cameras to export (default: none). Topics /<camera_id>/image/compressed and camera_info.
    :param time_field: per-point time property of the lidar point cloud files, in ns (None: no per-point time).
    :param intensity_field: intensity property of the lidar point cloud files (None: no intensity).
    :param storage: rosbag2 storage: sqlite3 (readable by every ROS2 distribution, eg. the GLIM humble image) or mcap
                    (needs the rosbag2_storage_mcap plugin on the ROS2 side, default from ROS2 Jazzy).
    :param compression: none or zstd (per message).
    :param force_overwrite_existing: delete an existing output bag.
    :return: number of messages written per topic.
    """
    if not has_rosbags:
        raise ImportError('exporting to ROS2 bags requires the python package rosbags (pip install rosbags)')
    storage_plugins = {'mcap': StoragePlugin.MCAP, 'sqlite3': StoragePlugin.SQLITE3}
    if storage not in storage_plugins:
        raise ValueError(f'unknown storage {storage}, expected one of {list(storage_plugins)}')
    if compression not in ('none', 'zstd'):
        raise ValueError(f'unknown compression {compression}, expected none or zstd')
    if path.exists(bag_dirpath):
        if not force_overwrite_existing:
            raise FileExistsError(f'{bag_dirpath} already exists, use force to overwrite')
        logger.info(f'deleting {bag_dirpath}')
        shutil.rmtree(bag_dirpath)

    logger.info(f'loading {kapture_dirpath}')
    skip_list = [kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures, kapture.Matches,
                 kapture.Points3d, kapture.Observations, kapture.RecordsWifi, kapture.RecordsBluetooth,
                 kapture.RecordsGnss, kapture.RecordsMagnetic, kapture.RecordsDepth, kapture.RecordsRange]
    kapture_data = csv.kapture_from_dir(kapture_dirpath, skip_list=skip_list)
    sensors = kapture_data.sensors
    rigs = kapture_data.rigs or kapture.Rigs()

    # select the sensors
    records_lidar = kapture_data.records_lidar or kapture.RecordsLidar()
    if lidar_ids is None:
        lidar_ids = sorted({sensor_id for _, sensor_id, _ in kapture.flatten(records_lidar)})
    for lidar_id in lidar_ids:
        if lidar_id not in sensors or sensors[lidar_id].sensor_type != kapture.SensorType.lidar.name:
            raise ValueError(f'{lidar_id} is not a lidar sensor of the kapture')
    if imu_ids is None:
        imu_ids = find_imu_ids(sensors)
    for imu_id in imu_ids:
        for suffix, sensor_type in ((ACCEL_SUFFIX, kapture.SensorType.accelerometer),
                                    (GYRO_SUFFIX, kapture.SensorType.gyroscope)):
            sensor = sensors.get(imu_id + suffix)
            if sensor is None or sensor.sensor_type != sensor_type.name:
                raise ValueError(f'IMU {imu_id}: no {sensor_type.name} sensor {imu_id + suffix} in the kapture')
    camera_ids = camera_ids or []
    for camera_id in camera_ids:
        if camera_id not in sensors or sensors[camera_id].sensor_type != kapture.SensorType.camera.name:
            raise ValueError(f'{camera_id} is not a camera sensor of the kapture')
    if not lidar_ids and not imu_ids and not camera_ids:
        raise ValueError('nothing to export: no lidar, IMU or camera selected')
    logger.info(f'lidars: {lidar_ids}, IMUs: {imu_ids}, cameras: {camera_ids}')

    # extrinsics, for the SLAM configuration
    for lidar_id in lidar_ids:
        for imu_id in imu_ids:
            pose = lidar_imu_extrinsics(rigs, lidar_id, imu_id + ACCEL_SUFFIX)
            if pose is None:
                logger.warning(f'{lidar_id} and {imu_id} are not in a common rig: no lidar-IMU extrinsics')
            else:
                tum = ' '.join(f'{v:.6f}' for v in pose_to_tum(pose))
                logger.info(f'T_lidar_imu ({lidar_id} <- {imu_id}), IMU pose in the lidar frame, '
                            f'[x y z qx qy qz qw]: {tum}')

    exporter = Ros2BagExporter()
    counts: Dict[str, int] = {}
    # metadata version 8: readable by ROS2 Humble (version 9 writes offered_qos_profiles as a list, Humble expects a
    # string and fails to open the bag), and by the newer distributions.
    writer = Writer(bag_dirpath, version=8, storage_plugin=storage_plugins[storage])
    if compression == 'zstd':
        writer.set_compression(CompressionMode.MESSAGE, CompressionFormat.ZSTD)
    with writer:
        streams: List[Iterator[Message]] = []
        # lidars
        for lidar_id in lidar_ids:
            connection = writer.add_connection(f'/{lidar_id}/points', exporter.PointCloud2.__msgtype__,
                                               typestore=exporter.typestore)
            records = sorted((timestamp, filename) for timestamp, sensor_id, filename
                             in kapture.flatten(records_lidar) if sensor_id == lidar_id)
            if not records:
                logger.warning(f'no records for lidar {lidar_id}')
            streams.append(_lidar_messages(exporter, connection, lidar_id, records, kapture_dirpath,
                                           time_field, intensity_field))
        # IMUs
        for imu_id in imu_ids:
            connection = writer.add_connection(f'/{imu_id}/imu', exporter.Imu.__msgtype__,
                                               typestore=exporter.typestore)
            streams.append(_imu_messages(exporter, connection, imu_id,
                                         kapture_data.records_accelerometer or kapture.RecordsAccelerometer(),
                                         kapture_data.records_gyroscope or kapture.RecordsGyroscope()))
        # cameras
        records_camera = kapture_data.records_camera or kapture.RecordsCamera()
        for camera_id in camera_ids:
            connection = writer.add_connection(f'/{camera_id}/image/compressed',
                                               exporter.CompressedImage.__msgtype__, typestore=exporter.typestore)
            info_connection = None
            if sensors[camera_id].camera_type.name in PINHOLE_CAMERA_INFO_MODELS:
                info_connection = writer.add_connection(f'/{camera_id}/camera_info',
                                                        exporter.CameraInfo.__msgtype__, typestore=exporter.typestore)
            else:
                logger.warning(f'camera {camera_id} ({sensors[camera_id].camera_type.name}): '
                               f'no CameraInfo for this camera model')
            records = sorted((timestamp, filename) for timestamp, sensor_id, filename
                             in kapture.flatten(records_camera) if sensor_id == camera_id)
            streams.append(_camera_messages(exporter, connection, info_connection, camera_id, sensors[camera_id],
                                            records, kapture_dirpath))
        # tf_static, at the first message time (needs the streams to be peeked: do it through a merge wrapper)
        tf_connection = None
        if rigs:
            tf_connection = writer.add_connection('/tf_static', exporter.TFMessage.__msgtype__,
                                                  typestore=exporter.typestore)

        merged = heapq.merge(*streams, key=lambda message: message[0])
        first = True
        hide_progress = logger.getEffectiveLevel() > logging.INFO
        for timestamp_ns, connection, data in tqdm(merged, disable=hide_progress, unit='msg'):
            if first and tf_connection is not None:
                writer.write(tf_connection, timestamp_ns, exporter.serialize(exporter.tf_message(timestamp_ns, rigs)))
                counts['/tf_static'] = 1
            first = False
            writer.write(connection, timestamp_ns, data)
            counts[connection.topic] = counts.get(connection.topic, 0) + 1

    logger.info('messages written: ' + ', '.join(f'{topic}: {n}' for topic, n in sorted(counts.items())))
    return counts


def _lidar_messages(exporter: Ros2BagExporter, connection, lidar_id: str, records: List[Tuple[int, str]],
                    kapture_dirpath: str, time_field: Optional[str], intensity_field: Optional[str]
                    ) -> Iterator[Message]:
    checked = False
    for timestamp_us, filename in records:
        vertices = read_ply(get_record_fullpath(kapture_dirpath, filename))
        if not checked:
            checked = True
            for name in ('x', 'y', 'z'):
                if name not in vertices.dtype.names:
                    raise ValueError(f'{filename}: no {name} property in the point cloud')
            if time_field and time_field not in vertices.dtype.names:
                logger.warning(f'{lidar_id}: no {time_field} property in {filename}, exporting without per-point '
                               f'time (the SLAM cannot deskew the scans)')
                time_field = None
            if intensity_field and intensity_field not in vertices.dtype.names:
                logger.warning(f'{lidar_id}: no {intensity_field} property in {filename}, exporting without '
                               f'intensity')
                intensity_field = None
        points, duration_ns = lidar_points_to_cloud(vertices, time_field, intensity_field)
        # kapture record time = last point, ROS header stamp = scan start
        timestamp_ns = us_to_ns(timestamp_us) - duration_ns
        yield timestamp_ns, connection, exporter.serialize(exporter.point_cloud2(timestamp_ns, lidar_id, points))


def _imu_messages(exporter: Ros2BagExporter, connection, imu_id: str,
                  records_accelerometer: kapture.RecordsAccelerometer, records_gyroscope: kapture.RecordsGyroscope
                  ) -> Iterator[Message]:
    accel_id, gyro_id = imu_id + ACCEL_SUFFIX, imu_id + GYRO_SUFFIX
    accel = sorted((timestamp, (record.x_accel, record.y_accel, record.z_accel))
                   for timestamp, sensor_id, record in kapture.flatten(records_accelerometer) if sensor_id == accel_id)
    gyro = sorted((timestamp, (record.x_speed, record.y_speed, record.z_speed))
                  for timestamp, sensor_id, record in kapture.flatten(records_gyroscope) if sensor_id == gyro_id)
    if not accel or not gyro:
        logger.warning(f'IMU {imu_id}: {len(accel)} accelerometer and {len(gyro)} gyroscope records, skipped')
        return
    gyro_timestamps = np.array([timestamp for timestamp, _ in gyro], dtype=np.float64)
    gyro_values = np.array([values for _, values in gyro], dtype=np.float64)
    gyro_by_timestamp = dict(gyro)
    nb_interpolated = 0
    for timestamp_us, accel_values in accel:
        gyro_values_at = gyro_by_timestamp.get(timestamp_us)
        if gyro_values_at is None:
            nb_interpolated += 1
            gyro_values_at = tuple(np.interp(float(timestamp_us), gyro_timestamps, gyro_values[:, i])
                                   for i in range(3))
        timestamp_ns = us_to_ns(timestamp_us)
        yield timestamp_ns, connection, exporter.serialize(
            exporter.imu(timestamp_ns, accel_id, accel_values, gyro_values_at))
    if nb_interpolated:
        logger.info(f'IMU {imu_id}: gyroscope interpolated at {nb_interpolated}/{len(accel)} accelerometer times')


def _camera_messages(exporter: Ros2BagExporter, connection, info_connection, camera_id: str, camera: kapture.Camera,
                     records: List[Tuple[int, str]], kapture_dirpath: str) -> Iterator[Message]:
    for timestamp_us, filename in records:
        extension = path.splitext(filename)[1].lower()
        image_format = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png'}.get(extension)
        if image_format is None:
            logger.warning(f'{filename}: not a jpeg or png image, skipped')
            continue
        with open(get_record_fullpath(kapture_dirpath, filename), 'rb') as f:
            data = f.read()
        timestamp_ns = us_to_ns(timestamp_us)
        if info_connection is not None:
            yield timestamp_ns, info_connection, exporter.serialize(
                exporter.camera_info(timestamp_ns, camera_id, camera))
        yield timestamp_ns, connection, exporter.serialize(
            exporter.compressed_image(timestamp_ns, camera_id, image_format, data))
