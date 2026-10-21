#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the kapture -> ROS2 bag export (lidar, IMU, cameras, rigs).
"""

import os.path as path
import tempfile
import unittest

import numpy as np
from PIL import Image

import kapture
import kapture.io.csv as kcsv
from kapture.io.ply import write_ply
from kapture.io.records import get_record_fullpath

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    from kapture.converter.ros_tools.export_ros2bag import kapture_to_ros2bag, lidar_imu_extrinsics, pose_to_tum
    has_rosbags = True
except ModuleNotFoundError:
    has_rosbags = False


def make_kapture(kapture_dirpath: str) -> kapture.Kapture:
    """ A small kapture: 1 lidar (2 scans), 1 IMU (accel + gyro), 1 pinhole camera (1 image), 1 rig. """
    sensors = kapture.Sensors()
    sensors['lidar0'] = kapture.Sensor(kapture.SensorType.lidar.name, ['os0-128'])
    sensors['imu_accel'] = kapture.Sensor(kapture.SensorType.accelerometer.name)
    sensors['imu_gyro'] = kapture.Sensor(kapture.SensorType.gyroscope.name)
    sensors['cam0'] = kapture.Camera(kapture.CameraType.PINHOLE, [8, 6, 5., 5., 4., 3.])
    sensors['omni'] = kapture.Camera(kapture.CameraType.OPENCV_OMNI, [8, 6, 5., 5., 0., 4., 3., 1., 0., 0., 0., 0.])
    rigs = kapture.Rigs()
    rigs['rig', 'lidar0'] = kapture.PoseTransform(r=[0., 1., 0., 0.], t=[-0.1, 0.2, -0.3])
    rigs['rig', 'imu_accel'] = kapture.PoseTransform(r=[0., 1., 0., 0.], t=[-0.1, 0.2, 0.3])
    rigs['rig', 'imu_gyro'] = rigs['rig', 'imu_accel']
    rigs['rig', 'cam0'] = kapture.PoseTransform(r=[0.5, 0.5, -0.5, 0.5], t=[0., 0., 0.])
    rigs['rig', 'omni'] = rigs['rig', 'cam0']
    # 2 scans, 100 ms each, kapture timestamp = last point time, per-point time in ns since the recording start
    records_lidar = kapture.RecordsLidar()
    for i, timestamp_us in enumerate([1_000_100_000, 1_000_200_000]):
        vertices = np.zeros(4, dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
                                      ('timestamps_0', '<f8'), ('intensities_0', '<u2')])
        vertices['x'] = [1., 2., 3., 4.]
        vertices['y'] = i
        vertices['z'] = -1.
        vertices['timestamps_0'] = i * 100_000_000 + np.array([0., 25e6, 50e6, 99_000_000.])
        vertices['intensities_0'] = [10, 20, 30, 40]
        filename = f'lidar0/{timestamp_us}.ply'
        write_ply(get_record_fullpath(kapture_dirpath, filename), vertices)
        records_lidar[timestamp_us, 'lidar0'] = filename
    # IMU: 4 accel records, gyro at the same times except the 3rd one (to be interpolated)
    records_accelerometer = kapture.RecordsAccelerometer()
    records_gyroscope = kapture.RecordsGyroscope()
    for k, timestamp_us in enumerate([1_000_000_000, 1_000_050_000, 1_000_100_000, 1_000_150_000]):
        records_accelerometer[timestamp_us, 'imu_accel'] = kapture.RecordAccelerometer(0.1 * k, 0.2, 9.8)
        gyro_timestamp_us = timestamp_us if k != 2 else timestamp_us + 10_000
        records_gyroscope[gyro_timestamp_us, 'imu_gyro'] = kapture.RecordGyroscope(0.01 * k, 0., 0.)
    # cameras
    records_camera = kapture.RecordsCamera()
    for camera_id in ('cam0', 'omni'):
        filename = f'{camera_id}/1000100000.jpg'
        image_filepath = get_record_fullpath(kapture_dirpath, filename)
        import os
        os.makedirs(path.dirname(image_filepath), exist_ok=True)
        Image.new('RGB', (8, 6), color=(10, 20, 30)).save(image_filepath, format='JPEG')
        records_camera[1_000_100_000, camera_id] = filename
    kapture_data = kapture.Kapture(sensors=sensors, rigs=rigs, records_lidar=records_lidar,
                                   records_accelerometer=records_accelerometer, records_gyroscope=records_gyroscope,
                                   records_camera=records_camera)
    kcsv.kapture_to_dir(kapture_dirpath, kapture_data)
    return kapture_data


@unittest.skipIf(not has_rosbags, "rosbags module is missing")
class TestExportRos2Bag(unittest.TestCase):

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.kapture_dirpath = path.join(self._tempdir.name, 'kapture')
        self.bag_dirpath = path.join(self._tempdir.name, 'bag')
        self.kapture_data = make_kapture(self.kapture_dirpath)
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def read_bag(self):
        """ :return: {topic: [(log time ns, message)]} """
        messages = {}
        with Reader(self.bag_dirpath) as reader:
            for connection, timestamp_ns, raw in reader.messages():
                message = self.typestore.deserialize_cdr(raw, connection.msgtype)
                messages.setdefault(connection.topic, []).append((timestamp_ns, message))
        return messages

    @staticmethod
    def stamp_ns(message) -> int:
        return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def test_export_lidar_imu(self):
        counts = kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath)
        self.assertEqual(counts, {'/lidar0/points': 2, '/imu/imu': 4, '/tf_static': 1})
        self.assertTrue(path.isfile(path.join(self.bag_dirpath, 'metadata.yaml')))
        messages = self.read_bag()
        self.assertEqual(set(messages), {'/lidar0/points', '/imu/imu', '/tf_static'})
        # lidar: header stamp = scan start = kapture timestamp - scan duration, t relative to the scan start
        (t0, scan0), (t1, scan1) = messages['/lidar0/points']
        self.assertEqual(t0, self.stamp_ns(scan0))
        self.assertEqual(t0, 1_000_100_000_000 - 99_000_000)
        self.assertEqual(t1, 1_000_200_000_000 - 99_000_000)
        self.assertEqual(scan0.header.frame_id, 'lidar0')
        self.assertEqual([f.name for f in scan0.fields], ['x', 'y', 'z', 'intensity', 't'])
        self.assertEqual(scan0.point_step, 20)
        self.assertEqual(scan0.width, 4)
        points = np.frombuffer(scan0.data, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                                  ('intensity', '<f4'), ('t', '<u4')])
        np.testing.assert_array_equal(points['x'], [1., 2., 3., 4.])
        np.testing.assert_array_equal(points['z'], -1.)
        np.testing.assert_array_equal(points['intensity'], [10., 20., 30., 40.])
        np.testing.assert_array_equal(points['t'], [0, 25_000_000, 50_000_000, 99_000_000])
        points1 = np.frombuffer(scan1.data, dtype=points.dtype)
        np.testing.assert_array_equal(points1['t'], [0, 25_000_000, 50_000_000, 99_000_000])
        np.testing.assert_array_equal(points1['y'], 1.)
        # IMU: one message per accelerometer record, gyroscope interpolated for the 3rd one
        imu = messages['/imu/imu']
        self.assertEqual([t for t, _ in imu], [1_000_000_000_000, 1_000_050_000_000, 1_000_100_000_000,
                                               1_000_150_000_000])
        self.assertEqual(imu[1][1].header.frame_id, 'imu_accel')
        self.assertAlmostEqual(imu[1][1].linear_acceleration.x, 0.1)
        self.assertAlmostEqual(imu[1][1].linear_acceleration.z, 9.8)
        self.assertAlmostEqual(imu[1][1].angular_velocity.x, 0.01)
        self.assertAlmostEqual(imu[2][1].angular_velocity.x, 0.01 + 0.01 * 50_000 / 60_000)  # interpolated
        self.assertEqual(imu[0][1].orientation_covariance[0], -1.)
        # tf_static: at the first message time, child pose in the parent = inverse of the kapture rig pose
        (tf_time, tf), = messages['/tf_static']
        self.assertEqual(tf_time, 1_000_000_000_000)
        transforms = {t.child_frame_id: t for t in tf.transforms}
        self.assertEqual(set(transforms), {'lidar0', 'imu_accel', 'imu_gyro', 'cam0', 'omni'})
        self.assertTrue(all(t.header.frame_id == 'rig' for t in tf.transforms))
        rig_from_lidar = self.kapture_data.rigs['rig', 'lidar0'].inverse()
        x, y, z, qx, qy, qz, qw = pose_to_tum(rig_from_lidar)
        lidar_tf = transforms['lidar0'].transform
        np.testing.assert_allclose([lidar_tf.translation.x, lidar_tf.translation.y, lidar_tf.translation.z],
                                   [x, y, z], atol=1e-9)
        np.testing.assert_allclose([lidar_tf.rotation.x, lidar_tf.rotation.y, lidar_tf.rotation.z,
                                    lidar_tf.rotation.w], [qx, qy, qz, qw], atol=1e-9)

    def test_lidar_imu_extrinsics(self):
        rigs = self.kapture_data.rigs
        pose = lidar_imu_extrinsics(rigs, 'lidar0', 'imu_accel')
        # a point at the IMU origin, seen in the lidar frame
        imu_origin_in_rig = rigs['rig', 'imu_accel'].inverse().t.ravel()
        imu_origin_in_lidar = rigs['rig', 'lidar0'].transform_points(imu_origin_in_rig[None, :])[0]
        np.testing.assert_allclose(pose.t.ravel(), imu_origin_in_lidar, atol=1e-9)
        np.testing.assert_allclose(pose.transform_points(np.zeros((1, 3)))[0], imu_origin_in_lidar, atol=1e-9)
        self.assertIsNone(lidar_imu_extrinsics(rigs, 'lidar0', 'unknown'))

    def test_export_cameras_and_options(self):
        counts = kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, lidar_ids=[], imu_ids=[],
                                    camera_ids=['cam0', 'omni'], storage='sqlite3', compression='zstd')
        self.assertEqual(counts, {'/cam0/image/compressed': 1, '/cam0/camera_info': 1,
                                  '/omni/image/compressed': 1, '/tf_static': 1})
        messages = self.read_bag()
        (t, image), = messages['/cam0/image/compressed']
        self.assertEqual(t, 1_000_100_000_000)
        self.assertEqual(image.format, 'jpeg')
        self.assertEqual(bytes(image.data[:2]), b'\xff\xd8')  # JPEG magic
        (_, info), = messages['/cam0/camera_info']
        self.assertEqual((info.width, info.height), (8, 6))
        self.assertEqual(info.distortion_model, 'plumb_bob')
        np.testing.assert_array_equal(info.k, [5., 0., 4., 0., 5., 3., 0., 0., 1.])
        np.testing.assert_array_equal(info.d, [0.] * 5)

    def test_without_time_and_intensity(self):
        kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, imu_ids=[], time_field=None,
                           intensity_field=None)
        messages = self.read_bag()
        (t0, scan0), _ = messages['/lidar0/points']
        self.assertEqual([f.name for f in scan0.fields], ['x', 'y', 'z'])
        self.assertEqual(t0, 1_000_100_000_000)  # no per-point time: stamp = kapture timestamp

    def test_errors(self):
        with self.assertRaises(ValueError):
            kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, lidar_ids=['cam0'])
        with self.assertRaises(ValueError):
            kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, imu_ids=['nope'])
        with self.assertRaises(ValueError):
            kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, lidar_ids=[], imu_ids=[])
        kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, imu_ids=[])
        with self.assertRaises(FileExistsError):
            kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, imu_ids=[])
        kapture_to_ros2bag(self.kapture_dirpath, self.bag_dirpath, imu_ids=[], force_overwrite_existing=True)


if __name__ == '__main__':
    unittest.main()
