#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the TUM trajectory import.
"""

import os.path as path
import tempfile
import unittest

import numpy as np
import quaternion

import kapture
import kapture.io.csv as kcsv
from kapture.converter.tum.import_tum import read_tum_trajectory, tum_to_trajectories, import_tum_trajectory


def make_kapture(kapture_dirpath: str) -> kapture.Kapture:
    sensors = kapture.Sensors()
    sensors['lidar0'] = kapture.Sensor(kapture.SensorType.lidar.name)
    sensors['imu_accel'] = kapture.Sensor(kapture.SensorType.accelerometer.name)
    rigs = kapture.Rigs()
    rigs['rig', 'lidar0'] = kapture.PoseTransform(r=[0., 1., 0., 0.], t=[-0.1, 0.2, -0.3])
    rigs['rig', 'imu_accel'] = kapture.PoseTransform(r=[1., 0., 0., 0.], t=[0., 0., 0.1])
    kapture_data = kapture.Kapture(sensors=sensors, rigs=rigs)
    kcsv.kapture_to_dir(kapture_dirpath, kapture_data)
    return kapture_data


def assert_pose_equal(test: unittest.TestCase, pose: kapture.PoseTransform, expected: kapture.PoseTransform):
    np.testing.assert_allclose(pose.t.ravel(), expected.t.ravel(), atol=1e-9)
    # q and -q are the same rotation
    q, e = quaternion.as_float_array(pose.r), quaternion.as_float_array(expected.r)
    if np.dot(q, e) < 0:
        q = -q
    np.testing.assert_allclose(q, e, atol=1e-9)


class TestImportTum(unittest.TestCase):

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.kapture_dirpath = path.join(self._tempdir.name, 'kapture')
        self.kapture_data = make_kapture(self.kapture_dirpath)
        self.tum_filepath = path.join(self._tempdir.name, 'traj.txt')
        # world_from_lidar poses: 90 deg about z, translation (1, 2, 3); then identity rotation
        self.world_from_lidar = [
            (1791356281469284, kapture.PoseTransform(r=[np.sqrt(0.5), 0., 0., np.sqrt(0.5)], t=[1., 2., 3.])),
            (1791356281569260, kapture.PoseTransform(r=[1., 0., 0., 0.], t=[1.5, 2., 3.])),
        ]
        with open(self.tum_filepath, 'w') as f:
            f.write('# timestamp tx ty tz qx qy qz qw\n\n')
            for timestamp_us, pose in self.world_from_lidar:
                t, r = pose.t.ravel(), pose.r
                f.write(f'{timestamp_us / 1e6:.6f} {t[0]} {t[1]} {t[2]} {r.x} {r.y} {r.z} {r.w}\n')

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_read(self):
        poses = read_tum_trajectory(self.tum_filepath)
        self.assertEqual(len(poses), 2)
        # microseconds of epoch seconds are kept exactly
        self.assertEqual([timestamp for timestamp, _ in poses], [t for t, _ in self.world_from_lidar])
        for (_, pose), (_, expected) in zip(poses, self.world_from_lidar):
            assert_pose_equal(self, pose, expected)

    def test_read_errors(self):
        with open(self.tum_filepath, 'w') as f:
            f.write('1.0 1 2 3 0 0 0\n')
        with self.assertRaises(ValueError):
            read_tum_trajectory(self.tum_filepath)

    def test_to_trajectories_device(self):
        poses = read_tum_trajectory(self.tum_filepath)
        trajectories = tum_to_trajectories(poses, 'lidar0')
        self.assertEqual(len(trajectories), 2)
        for timestamp_us, world_from_lidar in self.world_from_lidar:
            assert_pose_equal(self, trajectories[timestamp_us, 'lidar0'], world_from_lidar.inverse())
            # the lidar origin is at the TUM translation in the world
            origin = trajectories[timestamp_us, 'lidar0'].inverse().t.ravel()
            np.testing.assert_allclose(origin, world_from_lidar.t.ravel(), atol=1e-9)

    def test_to_trajectories_rig(self):
        poses = read_tum_trajectory(self.tum_filepath)
        rigs = self.kapture_data.rigs
        trajectories = tum_to_trajectories(poses, 'lidar0', rigs, 'rig')
        for timestamp_us, world_from_lidar in self.world_from_lidar:
            self.assertNotIn('lidar0', trajectories[timestamp_us])
            rig_from_world = trajectories[timestamp_us, 'rig']
            # moving the rig pose back to the lidar with the rig gives the TUM pose
            lidar_from_world = kapture.PoseTransform.compose([rigs['rig', 'lidar0'], rig_from_world])
            assert_pose_equal(self, lidar_from_world, world_from_lidar.inverse())
        # kapture.rigs_remove gives the same lidar poses
        lidar_trajectories = kapture.rigs_remove(trajectories, rigs)
        for timestamp_us, world_from_lidar in self.world_from_lidar:
            assert_pose_equal(self, lidar_trajectories[timestamp_us, 'lidar0'], world_from_lidar.inverse())
        with self.assertRaises(ValueError):
            tum_to_trajectories(poses, 'lidar0', rigs, 'unknown_rig')
        with self.assertRaises(ValueError):
            tum_to_trajectories(poses, 'unknown', rigs, 'rig')

    def test_import(self):
        trajectories = import_tum_trajectory(self.tum_filepath, self.kapture_dirpath, 'lidar0', 'rig')
        trajectories_filepath = kcsv.get_csv_fullpath(kapture.Trajectories, self.kapture_dirpath)
        self.assertTrue(path.isfile(trajectories_filepath))
        read_back = kcsv.trajectories_from_file(trajectories_filepath)
        self.assertEqual(set(read_back.key_pairs()), set(trajectories.key_pairs()))
        for timestamp_us, device_id in trajectories.key_pairs():
            assert_pose_equal(self, read_back[timestamp_us, device_id], trajectories[timestamp_us, device_id])
        with self.assertRaises(FileExistsError):
            import_tum_trajectory(self.tum_filepath, self.kapture_dirpath, 'lidar0', 'rig')
        import_tum_trajectory(self.tum_filepath, self.kapture_dirpath, 'lidar0', force_overwrite_existing=True)
        with self.assertRaises(ValueError):
            import_tum_trajectory(self.tum_filepath, self.kapture_dirpath, 'unknown', force_overwrite_existing=True)


if __name__ == '__main__':
    unittest.main()
