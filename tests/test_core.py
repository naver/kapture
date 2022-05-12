#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the kapture core module.
"""

import unittest
import numpy as np
import quaternion
from copy import deepcopy
from datetime import datetime
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.algo.compare import equal_trajectories, equal_rigs, equal_poses


# FLATTEN ##############################################################################################################
class TestFlatten(unittest.TestCase):
    def test_non_dict(self):
        test_values = 5
        expected_list = [(test_values,)]
        actual_list = list(kapture.flatten(test_values, is_sorted=False))
        actual_list_sorted = list(kapture.flatten(test_values, is_sorted=True))
        self.assertEqual(actual_list, expected_list)
        self.assertEqual(actual_list_sorted, expected_list)

        test_values = [5.0, 7.3, 6.02]
        expected_list = [(5.0,), (7.3,), (6.02,)]
        expected_list_sorted = [(5.0,), (6.02,), (7.3,)]
        actual_list = list(kapture.flatten(test_values, is_sorted=False))
        actual_list_sorted = list(kapture.flatten(test_values, is_sorted=True))
        self.assertEqual(actual_list, expected_list)
        self.assertEqual(actual_list_sorted, expected_list_sorted)

        test_values = kapture.RecordWifiSignal(frequency=2500, rssi=-1.0)
        expected_list = [(test_values,)]
        actual_list = list(kapture.flatten(test_values, is_sorted=False))
        actual_list_sorted = list(kapture.flatten(test_values, is_sorted=True))
        self.assertEqual(actual_list, expected_list)
        self.assertEqual(actual_list_sorted, expected_list)

    def test_simple_dict(self):
        test_values = {'a': 1, 'c': 3, 'b': 4, 'z': 0, 'y': 9, 'd': 5}
        expected_list = [('a', 1), ('c', 3), ('b', 4), ('z', 0), ('y', 9), ('d', 5)]
        expected_list_sorted = [('a', 1), ('b', 4), ('c', 3), ('d', 5), ('y', 9), ('z', 0)]
        actual_list = list(kapture.flatten(test_values, is_sorted=False))
        actual_list_sorted = list(kapture.flatten(test_values, is_sorted=True))
        self.assertEqual(actual_list, expected_list)
        self.assertEqual(actual_list_sorted, expected_list_sorted)

    def test_dict_of_list(self):
        test_values = {'a': [1, 4, 3], 'z': [1, 0], 'y': [9, 5]}
        expected_list = [('a', 1), ('a', 4), ('a', 3), ('z', 1), ('z', 0), ('y', 9), ('y', 5)]
        expected_list_sorted = [('a', 1), ('a', 3), ('a', 4), ('y', 5), ('y', 9), ('z', 0), ('z', 1)]
        actual_list = list(kapture.flatten(test_values, is_sorted=False))
        actual_list_sorted = list(kapture.flatten(test_values, is_sorted=True))
        self.assertEqual(actual_list, expected_list)
        self.assertEqual(actual_list_sorted, expected_list_sorted)

    def test_nested_dict(self):
        test_values = {'a': {'a': 1, 'c': 3, 'b': 4}, 'c': {'z': 0, 'y': 9, 'd': 5}, 'b': {'u': 7, 'a': 13, 'c': 1}}
        expected_list = [('a', 'a', 1), ('a', 'c', 3), ('a', 'b', 4),
                         ('c', 'z', 0), ('c', 'y', 9), ('c', 'd', 5),
                         ('b', 'u', 7), ('b', 'a', 13), ('b', 'c', 1)]
        expected_list_sorted = [('a', 'a', 1), ('a', 'b', 4), ('a', 'c', 3),
                                ('b', 'a', 13), ('b', 'c', 1), ('b', 'u', 7),
                                ('c', 'd', 5), ('c', 'y', 9), ('c', 'z', 0)]
        actual_list = list(kapture.flatten(test_values, is_sorted=False))
        actual_list_sorted = list(kapture.flatten(test_values, is_sorted=True))
        self.assertEqual(actual_list, expected_list)
        self.assertEqual(actual_list_sorted, expected_list_sorted)


# POSES ################################################################################################################
class TestPoseTransformConstruct(unittest.TestCase):

    def test_canonicals_list(self):
        expect_r = [1., 0., 0., 0.]
        expect_t = [0., 0., 0.]
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r
        actual_t = pose.t
        # check r
        self.assertIsInstance(actual_r, np.quaternion)
        self.assertListEqual(quaternion.as_float_array(
            actual_r).tolist(), expect_r)
        # check t
        self.assertIsInstance(actual_t, np.ndarray)
        self.assertTupleEqual(actual_t.shape, (3, 1))
        self.assertListEqual(actual_t.flatten().tolist(), expect_t)

    def test_canonicals_numpy(self):
        expect_r = np.array([1., 0., 0., 0.])
        expect_t = np.array([0., 0., 0.])
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r
        actual_t = pose.t
        # check r
        self.assertIsInstance(actual_r, np.quaternion)
        self.assertListEqual(quaternion.as_float_array(
            actual_r).tolist(), expect_r.tolist())
        # check t
        self.assertIsInstance(actual_t, np.ndarray)
        self.assertTupleEqual(actual_t.shape, (3, 1))
        self.assertListEqual(actual_t.flatten().tolist(), expect_t.tolist())

    def test_canonicals_quaternion(self):
        expect_r = quaternion.from_float_array([1., 0., 0., 0.])
        expect_t = np.array([0., 0., 0.])
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r
        actual_t = pose.t
        # check r
        self.assertIsInstance(actual_r, np.quaternion)
        self.assertListEqual(quaternion.as_float_array(
            actual_r).tolist(), quaternion.as_float_array(expect_r).tolist())
        # check t
        self.assertIsInstance(actual_t, np.ndarray)
        self.assertTupleEqual(actual_t.shape, (3, 1))
        self.assertListEqual(actual_t.flatten().tolist(), expect_t.tolist())

    def test_none_list(self):
        expect_r = [None] * 4
        expect_t = [None] * 3
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r
        actual_t = pose.t
        # check r
        self.assertEqual(actual_r, None)
        # check t
        self.assertEqual(actual_t, None)

    def test_none(self):
        expect_r = None
        expect_t = None
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r
        actual_t = pose.t
        # check r
        self.assertEqual(actual_r, None)
        # check t
        self.assertEqual(actual_t, None)


class TestPoseTransformRaw(unittest.TestCase):

    def test_canonicals_list(self):
        expect_r = [1., 0., 0., 0.]
        expect_t = [0., 0., 0.]
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r_raw
        actual_t = pose.t_raw
        self.assertListEqual(actual_r, expect_r)
        self.assertListEqual(actual_t, expect_t)

    def test_none(self):
        expect_r = None
        expect_t = None
        pose = kapture.PoseTransform(expect_r, expect_t)
        actual_r = pose.r_raw
        actual_t = pose.t_raw
        self.assertEqual(actual_r, None)
        self.assertEqual(actual_t, None)


class TestPoseTransformApply(unittest.TestCase):
    def setUp(self):
        self.expected_points3d = np.arange(3 * 10).reshape(-1, 3)

    def test_identity(self):
        # default construct
        pose_identity = kapture.PoseTransform()
        actual_points3d = pose_identity.transform_points(
            self.expected_points3d)
        self.assertTrue(np.all(actual_points3d == self.expected_points3d))
        # explicit identity
        pose_identity = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        actual_points3d = pose_identity.transform_points(
            self.expected_points3d)
        self.assertTrue(np.all(actual_points3d == self.expected_points3d))

    def test_translation(self):
        pose_translate = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 10])
        actual_points3d = pose_translate.transform_points(
            self.expected_points3d)
        diff = (actual_points3d - self.expected_points3d)
        self.assertTrue(np.all(diff[:, 2] == 10))
        self.assertTrue(np.all(diff[:, 0:2] == 0))

    def test_rotation(self):
        r = quaternion.from_rotation_vector(
            [0, 0, np.pi])  # rotate 180° around Z axis
        # print(quaternion.as_rotation_matrix(r))
        pose_rotate = kapture.PoseTransform(r=r, t=[0, 0, 0])
        actual_points3d = pose_rotate.transform_points(self.expected_points3d)
        self.assertTrue(np.all(np.isclose(
            actual_points3d[:, 0:2], - self.expected_points3d[:, 0:2])))  # X, Y opposed
        self.assertTrue(np.all(np.isclose(
            actual_points3d[:, 2], self.expected_points3d[:, 2])))  # Z untouched


class TestPoseEquality(unittest.TestCase):
    def test_equal(self):
        pose_1 = kapture.PoseTransform(r=[1, 10, 0, 0], t=[0, 0, 10])
        pose_2 = kapture.PoseTransform(r=[1, 10, 0, 0], t=[0, 0, 10])
        self.assertEqual(pose_1, pose_2)
        self.assertNotEqual(pose_1, kapture.PoseTransform(t=None))
        self.assertNotEqual(pose_1, kapture.PoseTransform(r=None))
        pose_2 = kapture.PoseTransform(r=[1, 10, 0, 0], t=[0, 0.000009, 10])
        self.assertEqual(pose_1, pose_2)
        pose_2 = kapture.PoseTransform(r=[1, 10, 0, 0], t=[0, 0.22, 10])
        self.assertNotEqual(pose_1, pose_2)
        pose_2 = kapture.PoseTransform(r=[1, 10, 0, 0.011], t=[0, 0.0, 10])
        self.assertNotEqual(pose_1, pose_2)
        pose_2 = kapture.PoseTransform(r=[1, 10, 0, 0.01], t=[0, 0.0, 10])
        self.assertEqual(pose_1, pose_2)


# SENSOR ###############################################################################################################
class TestSensor(unittest.TestCase):
    def test_init(self):
        # test bare minimum
        sensor = kapture.Sensor('unknown', [])
        self.assertEqual(sensor.name, None)
        self.assertEqual(sensor.sensor_type, 'unknown')
        self.assertListEqual(sensor.sensor_params, [])

        # test typical camera
        sensor_name = 'GOPRO_FUSION'
        sensor_type = kapture.SensorType.camera.name
        #                 SIMPLE_PINHOLE,   w,   h,   f,  cx,  cy
        sensor_params = ['SIMPLE_PINHOLE', 640, 480, 100, 320, 240]
        sensor = kapture.Sensor(sensor_type, sensor_params, sensor_name)
        self.assertEqual(sensor.name, sensor_name)
        self.assertEqual(sensor.sensor_type, sensor_type)
        self.assertListEqual(sensor.sensor_params, [i for i in sensor_params])
        self.assertIsInstance(sensor.__repr__(), str)

        sensor = kapture.Camera(sensor_params[0], sensor_params[1:], sensor_name)
        self.assertEqual(sensor.name, sensor_name)
        self.assertEqual(sensor.sensor_type, sensor_type)
        self.assertEqual(sensor.camera_type, kapture.CameraType.SIMPLE_PINHOLE)
        self.assertListEqual(sensor.sensor_params, [str(i) for i in sensor_params])
        self.assertListEqual(sensor.camera_params, [float(i) for i in sensor_params[1:]])
        self.assertIsInstance(sensor.__repr__(), str)

    def test_update(self):
        pass
        # TODO
        # sensor = kapture.Sensor('unknown', [])
        # sensor.name = 'toto'


class TestCameraModel(unittest.TestCase):
    def test_model_params_fully_set(self):
        for sensor_model in kapture.CameraType:
            self.assertIn(sensor_model, kapture.CAMERA_TYPE_PARAMS_COUNT)
            self.assertIn(sensor_model.value,
                          kapture.CAMERA_TYPE_PARAMS_COUNT_FROM_NAME)

    def test_dict(self):
        model = kapture.CameraType.FULL_OPENCV
        self.assertEqual(model.value, "FULL_OPENCV")
        self.assertEqual(
            kapture.CAMERA_TYPE_PARAMS_COUNT[model], kapture.CAMERA_TYPE_PARAMS_COUNT_FROM_NAME[model.value])
        self.assertEqual(kapture.CAMERA_TYPE_PARAMS_COUNT[model], 14)


# SENSORS ##############################################################################################################
class TestSensors(unittest.TestCase):
    def test_init(self):
        sensors = kapture.Sensors()
        self.assertEqual(0, len(sensors))
        sensors['cam0'] = kapture.Sensor('unknown', [])
        self.assertEqual(1, len(sensors))
        self.assertIn('cam0', sensors)
        self.assertIn('unknown', sensors['cam0'].sensor_type)

    def test_type_checking(self):
        sensors = kapture.Sensors()
        invalid_sensor_id = tuple('a', )
        valid_sensor_id = 'cam0'
        invalid_sensor = int(0)
        valid_sensor = kapture.Sensor(kapture.SensorType.camera.name)
        self.assertRaises(TypeError, sensors.__setitem__, valid_sensor_id, invalid_sensor)
        self.assertRaises(TypeError, sensors.__setitem__, invalid_sensor_id, valid_sensor)
        self.assertRaises(TypeError, sensors.__setitem__, invalid_sensor_id, invalid_sensor)


# RIGS #################################################################################################################
class TestRigs(unittest.TestCase):
    def test_init(self):
        rigs = kapture.Rigs()
        rigs['rig0', 'cam0'] = kapture.PoseTransform()
        rigs['rig0', 'cam1'] = kapture.PoseTransform()
        rigs['rig1'] = {
            'cam2': kapture.PoseTransform(),
            'cam3': kapture.PoseTransform()
        }
        self.assertEqual(2, len(rigs))
        self.assertIn('rig0', rigs)
        self.assertIn('rig1', rigs)
        self.assertIn('cam0', rigs['rig0'])
        self.assertNotIn('cam0', rigs['rig1'])

    def test_type_checking(self):
        rigs = kapture.Rigs()
        valid_id, valid_pose = 'rig0', kapture.PoseTransform()
        invalid_id, invalid_pose = float(0), 'rig'

        self.assertRaises(TypeError, rigs.__setitem__, valid_id, valid_pose)
        self.assertRaises(TypeError, rigs.__setitem__, (valid_id, invalid_id), valid_pose)
        self.assertRaises(TypeError, rigs.__setitem__, (invalid_id, valid_id), valid_pose)
        self.assertRaises(TypeError, rigs.__setitem__, (valid_id, valid_id), invalid_pose)


# TRAJECTORIES #########################################################################################################
class TestTrajectories(unittest.TestCase):
    def test_init(self):
        timestamp1 = 0
        timestamp2 = 1
        device_id = 'cam0'
        pose = kapture.PoseTransform()
        traj = kapture.Trajectories()
        # pair assignment
        traj[(timestamp1, device_id)] = pose
        # dict assignment
        traj[timestamp2] = {device_id: pose}
        self.assertEqual(2, len(traj))
        self.assertEqual(traj[timestamp1],
                         traj[timestamp2])
        self.assertEqual(traj[(timestamp1, device_id)],
                         traj[(timestamp2, device_id)])

        # test __contains__
        self.assertIn(timestamp1, traj)
        self.assertIn((timestamp1, device_id), traj)
        self.assertIn(timestamp2, traj)
        self.assertIn((timestamp1, device_id), traj)

        self.assertNotIn((timestamp1, 'cam1'), traj)
        self.assertNotIn((2, device_id), traj)

    def test_type_checking(self):
        traj = kapture.Trajectories()
        valid_ts, valid_id, valid_pose = 0, 'cam0', kapture.PoseTransform()
        invalid_ts, invalid_id, invalid_pose = '0', float(0), 'pose'
        self.assertRaises(TypeError, traj.__setitem__, (invalid_ts, valid_id), valid_pose)
        self.assertRaises(TypeError, traj.__setitem__, (valid_ts, invalid_id), valid_pose)
        self.assertRaises(TypeError, traj.__setitem__, (valid_ts, valid_id), invalid_pose)
        self.assertRaises(TypeError, traj.__setitem__, (invalid_ts, invalid_id), invalid_pose)

        self.assertRaises(TypeError, traj.__setitem__, invalid_ts, {valid_id: valid_pose})
        self.assertRaises(TypeError, traj.__setitem__, valid_ts, {invalid_id: valid_pose})
        self.assertRaises(TypeError, traj.__setitem__, valid_ts, {valid_id: invalid_pose})
        self.assertRaises(TypeError, traj.__setitem__, invalid_ts, valid_pose)

        self.assertRaises(TypeError, traj.__contains__, invalid_ts, valid_id)
        self.assertRaises(TypeError, traj.__contains__, valid_ts, invalid_id)
        self.assertRaises(TypeError, traj.__contains__, invalid_ts, invalid_id)

        self.assertRaises(TypeError, traj.__delitem__, invalid_ts)
        self.assertRaises(TypeError, traj.__delitem__, (valid_ts, invalid_id))

    def test_inverse(self):
        pose1 = kapture.PoseTransform()
        pose2 = kapture.PoseTransform(
            r=[-0.09465784241766664, 0.4240600207396319, -0.6602839272564094, 0.6125664478512957],
            t=[-81525.97101792274, 6362571.7626688825, 216599.32985438255]
        )
        pose3 = kapture.PoseTransform(
            r=[-0.07509999120329255, 0.428687830189451, -0.6630694170354837, 0.6090368344187961],
            t=[49418.858719801996, 6364643.125801517, 157364.12971148116]
        )
        traj = kapture.Trajectories()
        traj[(0, 'dummy1')] = pose1
        traj[(1, 'dummy1')] = pose2
        traj[(0, 'dummy2')] = pose2
        traj[(1, 'dummy2')] = pose3
        traj_inv = traj.inverse()

        self.assertEqual(traj.key_pairs(), traj_inv.key_pairs())
        self.assertTrue(equal_poses(traj_inv[0, 'dummy1'], pose1))  # identity
        self.assertTrue(equal_poses(traj_inv[1, 'dummy1'], pose2.inverse()))  #
        self.assertTrue(equal_poses(traj_inv[1, 'dummy2'], pose3.inverse()))  #

    def test_rig_remove(self):
        rigs = kapture.Rigs()
        rigs['rig0', 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[100, 0, 0])
        rigs['rig0', 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[-100, 0, 0])
        trajectories = kapture.Trajectories()
        trajectories[0, 'rig0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        trajectories[1, 'rig0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 10])
        trajectories[2, 'rig0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 20])
        trajectories_ = kapture.rigs_remove(trajectories, rigs)
        # timestamps should be unchanged
        self.assertEqual(trajectories_.keys(), trajectories.keys())
        self.assertNotEqual(trajectories_.key_pairs(), trajectories.key_pairs())
        self.assertEqual(len(trajectories_.key_pairs()), len(trajectories.key_pairs()) * len(rigs.key_pairs()))
        self.assertIn((0, 'cam0'), trajectories_.key_pairs())
        self.assertIn((0, 'cam1'), trajectories_.key_pairs())
        self.assertIn((2, 'cam0'), trajectories_.key_pairs())
        self.assertIn((2, 'cam1'), trajectories_.key_pairs())
        self.assertAlmostEqual(trajectories_[2, 'cam1'].t_raw, [-100.0, 0.0, 20.0])
        self.assertAlmostEqual(trajectories_[2, 'cam1'].r_raw, [1.0, 0.0, 0.0, 0.0])

    def test_remove(self):
        trajectories = kapture.Trajectories()
        trajectories[0, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        trajectories[1, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 10])
        trajectories[2, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 20])
        trajectories[2, 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[10, 0, 20])
        self.assertEqual(len(trajectories), 3)
        self.assertEqual(len(trajectories[2]), 2)

        del trajectories[2, 'cam0']
        self.assertEqual(len(trajectories), 3)
        self.assertEqual(len(trajectories[2]), 1)

        del trajectories[1]
        self.assertEqual(len(trajectories), 2)
        del trajectories[2, 'cam1']
        self.assertEqual(len(trajectories), 1)

    def test_timestamps_list(self):
        trajectories = kapture.Trajectories()
        trajectories[2, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 20])
        trajectories[0, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        trajectories[2, 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[10, 0, 20])
        self.assertEqual(trajectories.timestamps_sorted_list(), [0, 2])
        trajectories[1, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 10])
        self.assertEqual(len(trajectories), 3)
        self.assertEqual(trajectories.timestamps_sorted_list(), [0, 1, 2])

        del trajectories[2, 'cam0']
        self.assertEqual(trajectories.timestamps_sorted_list(), [0, 1, 2])

        del trajectories[1]
        self.assertEqual(trajectories.timestamps_sorted_list(), [0, 2])
        del trajectories[2, 'cam1']
        self.assertEqual(trajectories.timestamps_sorted_list(), [0, ])

    def test_timestamps_length(self):
        trajectories = kapture.Trajectories()
        self.assertEqual(trajectories.timestamp_length(), -1)
        trajectories[1614362592378, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 20])
        trajectories[1614362592634, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        trajectories[1614362592378, 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[10, 0, 20])
        trajectories[1614362593123, 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[10, 0, 30])
        self.assertEqual(trajectories.timestamp_length(), 13)
        # Check that if we have timestamps of different precision, we can not compute a common length
        trajectories[1614362594, 'lidar0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        self.assertEqual(trajectories.timestamp_length(), -1)

    def test_pose_interpolation(self):
        trajectories = kapture.Trajectories()
        trajectories[1614362592000, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 0])
        trajectories[1614362592000, 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 10, 0])
        trajectories[1614362592500, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 10])
        trajectories[1614362593000, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 20])
        trajectories[1614362593000, 'cam1'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 10, 20])
        trajectories[1614362593500, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 30])
        trajectories[1614362594000, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 40])
        trajectories[1614362594500, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 50])
        trajectories[1614362595000, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 60])
        trajectories[1614362595500, 'cam0'] = kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 70])
        self.assertEqual(trajectories.timestamp_length(), 13)
        pose = trajectories.intermediate_pose(1614362592500, 'cam2', 1000000)
        self.assertIsNone(pose, "unknown device")
        pose = trajectories.intermediate_pose(1614362593000, 'cam0', 1000000)
        self.assertEqual(pose, kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 20]), "existing pose")
        pose = trajectories.intermediate_pose(1614362500000, 'cam0', 1000000)
        self.assertIsNone(pose, "time too far in past")
        pose = trajectories.intermediate_pose(1614362600000, 'cam0', 1000000)
        self.assertIsNone(pose, "time too far in future")
        pose = trajectories.intermediate_pose(1614362595250, 'cam0', 1000000)
        self.assertEqual(pose, kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 0, 65]))
        pose = trajectories.intermediate_pose(1614362592500, 'cam1', 1000000)
        self.assertEqual(pose, kapture.PoseTransform(r=[1, 0, 0, 0], t=[0, 10, 10]))
        pose = trajectories.intermediate_pose(1614362595250, 'cam1', 1000000)
        self.assertIsNone(pose, "not enough pose for cam1")


# REMOVE/RESTORE RIGS in TRAJECTORIES ##################################################################################
class TestTrajectoriesRig(unittest.TestCase):
    def setUp(self):
        self._rigs = kapture.Rigs()
        looking_not_straight = quaternion.from_rotation_vector([0, np.deg2rad(5.), 0])
        self._rigs['rig', 'phone'] = kapture.PoseTransform()  # do a nested rig
        self._rigs['phone', 'cam1'] = kapture.PoseTransform(t=[-10, 0, 0], r=looking_not_straight).inverse()
        self._rigs['phone', 'cam2'] = kapture.PoseTransform(t=[+10, 0, 0], r=looking_not_straight.inverse()).inverse()
        self._trajectories_rigs = kapture.Trajectories()
        for timestamp, ratio in enumerate(np.linspace(0., 1., num=8)):
            looking_around = quaternion.from_rotation_vector([0, np.deg2rad(360. * ratio), 0])
            self._trajectories_rigs[timestamp, 'rig'] = kapture.PoseTransform(t=[0, 0, -100.], r=looking_around)

        self._trajectories_cams = kapture.Trajectories()
        for timestamp, rig_id, pose_rig_from_world in kapture.flatten(self._trajectories_rigs, is_sorted=True):
            for rig_id2, cam_id, pose_cam_from_rig in kapture.flatten(self._rigs):
                if cam_id == 'phone':
                    continue
                pose_cam_from_world = kapture.PoseTransform.compose([pose_cam_from_rig, pose_rig_from_world])
                self._trajectories_cams[timestamp, cam_id] = pose_cam_from_world

    def test_rig_remove(self):
        trajectories_wo_rigs = kapture.rigs_remove(self._trajectories_rigs, self._rigs)
        self.assertTrue(equal_trajectories(trajectories_wo_rigs, self._trajectories_cams))

    def test_rig_remove_inplace(self):
        trajectories = deepcopy(self._trajectories_rigs)
        rigs = deepcopy(self._rigs)
        kapture.rigs_remove_inplace(trajectories, rigs)
        self.assertTrue(equal_trajectories(trajectories, self._trajectories_cams))
        self.assertTrue(equal_rigs(rigs, self._rigs))

    def test_rig_remove_inplace_consistency(self):
        # compare inplace and not inplace
        trajectories_inplace = deepcopy(self._trajectories_rigs)
        rigs_inplace = deepcopy(self._rigs)
        kapture.rigs_remove_inplace(trajectories_inplace, rigs_inplace)
        trajectories_not_inplace = kapture.rigs_remove(self._trajectories_rigs, self._rigs)
        self.assertTrue(equal_trajectories(trajectories_inplace, trajectories_not_inplace))
        # make sure rigs are untouched.
        self.assertTrue(equal_rigs(rigs_inplace, self._rigs))

    def test_rig_recover(self):
        trajectories_w_rigs = kapture.rigs_recover(self._trajectories_cams, self._rigs)
        self.assertTrue(equal_trajectories(trajectories_w_rigs, self._trajectories_rigs))

    def test_rig_recover_inplace(self):
        trajectories = deepcopy(self._trajectories_cams)
        rigs = deepcopy(self._rigs)
        kapture.rigs_recover_inplace(trajectories, rigs)
        self.assertTrue(equal_trajectories(trajectories, self._trajectories_rigs))
        self.assertTrue(equal_rigs(rigs, self._rigs))

    def test_rig_recover_inplace_consistency(self):
        # compare inplace and not inplace
        trajectories_inplace = deepcopy(self._trajectories_cams)
        rigs_inplace = deepcopy(self._rigs)
        kapture.rigs_recover_inplace(trajectories_inplace, rigs_inplace)
        trajectories_not_inplace = kapture.rigs_recover(self._trajectories_cams, self._rigs)
        self.assertTrue(equal_trajectories(trajectories_inplace, trajectories_not_inplace))
        # x = [pose.inverse().t_raw
        #      for _, _, pose in kapture.flatten(rigs_inplace, is_sorted=True)]
        # x = np.array(x)
        # import matplotlib.pyplot as plt
        # plt.plot(x[:, 0], x[:, 2], 'x--')
        # plt.show()

    def test_rig_sensors_ids(self):
        sensors_ids = self._trajectories_cams.sensors_ids
        self.assertEqual(len(sensors_ids), 2)
        self.assertIn('cam2', sensors_ids)


# RECORDS ##############################################################################################################
class TestRecords(unittest.TestCase):
    def test_init_camera(self):
        timestamp0, timestamp1 = 0, 1
        device_id0, device_id1 = 'cam0', 'cam1'
        record_cam0_image0 = 'cam0/image000.jpg'
        record_cam0_image1 = 'cam0/image001.jpg'
        record_cam1_image0 = 'cam1/image000.jpg'
        record_cam1_image1 = 'cam1/image001.jpg'
        # Test insertions
        records_camera = kapture.RecordsCamera()
        records_camera[timestamp0, device_id0] = record_cam0_image0
        kapture_data = kapture.Kapture(records_camera=records_camera)
        self.assertEqual(1, len(kapture_data.records_camera.keys()))
        self.assertEqual(1, len(kapture_data.records_camera.key_pairs()))
        self.assertEqual(record_cam0_image0, kapture_data.records_camera.data_list()[0])
        self.assertIn(timestamp0, kapture_data.records_camera)
        self.assertIn(device_id0, kapture_data.records_camera[timestamp0])
        self.assertIn((timestamp0, device_id0), kapture_data.records_camera)
        self.assertEqual(record_cam0_image0, kapture_data.records_camera[timestamp0, device_id0])
        records_camera[timestamp1, device_id0] = record_cam0_image1
        self.assertEqual(2, len(kapture_data.records_camera.keys()))
        self.assertEqual(2, len(kapture_data.records_camera.key_pairs()))
        self.assertEqual(2, len(kapture_data.records_camera.data_list()))
        self.assertEqual(record_cam0_image1, kapture_data.records_camera.data_list()[-1])
        kapture_data.records_camera[timestamp0][device_id1] = record_cam1_image0
        self.assertEqual(2, len(kapture_data.records_camera.keys()))
        self.assertEqual(3, len(kapture_data.records_camera.key_pairs()))
        self.assertEqual(3, len(kapture_data.records_camera.data_list()))
        self.assertEqual(record_cam0_image1, kapture_data.records_camera.data_list()[-1])
        records_camera[timestamp1][device_id1] = record_cam1_image1
        self.assertEqual(2, len(kapture_data.records_camera.keys()))
        self.assertEqual(4, len(kapture_data.records_camera.key_pairs()))
        self.assertEqual(4, len(kapture_data.records_camera.data_list()))
        self.assertEqual(record_cam1_image1, kapture_data.records_camera.data_list()[-1])
        self.assertEqual(record_cam0_image0, kapture_data.records_camera[timestamp0, device_id0])
        self.assertEqual(record_cam1_image0, kapture_data.records_camera[timestamp0, device_id1])
        self.assertEqual(record_cam0_image1, kapture_data.records_camera[timestamp1, device_id0])
        self.assertEqual(record_cam1_image1, kapture_data.records_camera[timestamp1, device_id1])

        self.assertNotIn((timestamp1, 'cam2'), kapture_data.records_camera)
        self.assertNotIn((2, device_id0), kapture_data.records_camera)
        self.assertEqual(kapture_data.records_camera.sensors_ids, {device_id0, device_id1})

        # Test deletion
        del kapture_data.records_camera[(timestamp0, device_id0)]
        self.assertEqual(2, len(kapture_data.records_camera.keys()))
        self.assertEqual(3, len(kapture_data.records_camera.key_pairs()))
        self.assertEqual(3, len(kapture_data.records_camera.data_list()))
        del kapture_data.records_camera[(timestamp0, device_id1)]
        self.assertEqual(1, len(kapture_data.records_camera.keys()))
        self.assertEqual(2, len(kapture_data.records_camera.key_pairs()))
        self.assertEqual(2, len(kapture_data.records_camera.data_list()))

    def test_init_lidar(self):
        records_lidar = kapture.RecordsLidar()
        self.assertIsNotNone(records_lidar, "Records Lidar created")

    def test_init_wifi(self):
        records_wifi = kapture.RecordsWifi()
        timestamp0, timestamp1 = 0, 1
        device_id0, device_id1 = 'AC01324954_WIFI', 'AC01324955_WIFI'
        bssid, ssid = '68:72:51:80:52:df', 'M1X_PicoM2'
        rssi = -33
        freq = 2417
        scan_time_start, scan_time_end = 1555398770280, 1555398770290
        # assign

        wifi_data = kapture.RecordWifi({bssid: kapture.RecordWifiSignal(ssid=ssid, rssi=rssi, frequency=freq,
                                                                        scan_time_start=scan_time_start,
                                                                        scan_time_end=scan_time_end)})
        records_wifi[timestamp0, device_id0] = wifi_data
        kapture_data = kapture.Kapture(records_wifi=records_wifi)
        self.assertEqual(1, len(kapture_data.records_wifi.keys()))
        self.assertEqual(1, len(kapture_data.records_wifi.key_pairs()))
        self.assertIn(timestamp0, kapture_data.records_wifi)
        self.assertIn(device_id0, kapture_data.records_wifi[timestamp0])
        self.assertIn((timestamp0, device_id0), kapture_data.records_wifi)
        self.assertEqual(wifi_data, kapture_data.records_wifi[timestamp0, device_id0])
        kapture_data.records_wifi[timestamp1, device_id1] = wifi_data
        self.assertEqual(2, len(kapture_data.records_wifi.keys()))
        self.assertEqual(2, len(kapture_data.records_wifi.key_pairs()))
        kapture_data.records_wifi[timestamp0][device_id1] = wifi_data
        self.assertEqual(2, len(kapture_data.records_wifi.keys()))
        self.assertEqual(3, len(kapture_data.records_wifi.key_pairs()))

    def test_init_gnss_epsg(self):
        gps_id1, gps_id2 = 'gps1', 'gps2'
        y, x, z = 51.388920, 30.099134, 15.0
        unix_ts = int(datetime(year=1986, month=4, day=26).timestamp())

        records_gnss = kapture.RecordsGnss()
        records_gnss[0, gps_id1] = kapture.RecordGnss(x + 0, y + 0, z + 0, unix_ts + 0, 9.)
        records_gnss[1, gps_id1] = kapture.RecordGnss(x + 1, y + 1, z + 1, unix_ts + 1, 2.)
        records_gnss[1, gps_id2] = kapture.RecordGnss(x + 2, y + 2, z + 2, unix_ts + 2, 2.)
        records_gnss[2] = {gps_id1: kapture.RecordGnss(x + 3, y + 3, z + 3, unix_ts + 3, 0.)}

        self.assertEqual(3, len(records_gnss))
        self.assertEqual(4, len(sorted(kapture.flatten(records_gnss))))
        self.assertIn((0, gps_id1), records_gnss)
        self.assertEqual(30.099134, records_gnss[0, gps_id1].x)
        self.assertEqual(51.388920, records_gnss[0, gps_id1].y)
        self.assertEqual(15., records_gnss[0, gps_id1].z)
        self.assertEqual(9., records_gnss[0, gps_id1].dop)

    def test_type_checking(self):
        records_camera = kapture.RecordsCamera()
        valid_ts, valid_id, valid_record = 0, 'cam0', 'cam0/image0.jpg'
        invalid_ts, invalid_id, invalid_record = '0', float(0), kapture.PoseTransform()
        self.assertRaises(TypeError, records_camera.__setitem__, (invalid_ts, valid_id), valid_record)
        self.assertRaises(TypeError, records_camera.__setitem__, (valid_ts, invalid_id), valid_record)
        self.assertRaises(TypeError, records_camera.__setitem__, (valid_ts, valid_id), invalid_record)
        self.assertRaises(TypeError, records_camera.__setitem__, (invalid_ts, invalid_id), invalid_record)

        self.assertRaises(TypeError, records_camera.__setitem__, invalid_ts, {valid_id: valid_record})
        self.assertRaises(TypeError, records_camera.__setitem__, valid_ts, {invalid_id: valid_record})
        self.assertRaises(TypeError, records_camera.__setitem__, valid_ts, {valid_id: invalid_record})
        self.assertRaises(TypeError, records_camera.__setitem__, invalid_ts, valid_record)

        self.assertRaises(TypeError, records_camera.__contains__, invalid_ts, valid_id)
        self.assertRaises(TypeError, records_camera.__contains__, valid_ts, invalid_id)
        self.assertRaises(TypeError, records_camera.__contains__, invalid_ts, invalid_id)

        self.assertRaises(TypeError, records_camera.__delitem__, invalid_ts)
        self.assertRaises(TypeError, records_camera.__delitem__, (valid_ts, invalid_id))


# Keypoints ############################################################################################################
class TestKeypoints(unittest.TestCase):
    def test_init_keypoints_sift(self):
        keypoints = kapture.Keypoints('SIFT', float, 4,
                                      ['a/a.jpg', 'b/b.jpg', 'c/c.jpg', 'c/c.jpg'])
        self.assertEqual('SIFT', keypoints.type_name)
        self.assertEqual(3, len(keypoints))
        self.assertIn('a/a.jpg', keypoints)

    def test_init_keypoints_unknown(self):
        keypoints = kapture.Keypoints('UNKNOWN', int, 64, ['a/a.jpg'])
        self.assertEqual('UNKNOWN', keypoints.type_name)


# Descriptors ##########################################################################################################
class TestDescriptors(unittest.TestCase):
    def test_init_descriptors_unknown(self):
        descriptors = kapture.Descriptors('R2D2', float, 64,
                                          'R2D2', 'L2',
                                          ['a/a.jpg', 'b/b.jpg', 'c/c.jpg', 'c/c.jpg'])
        self.assertEqual('R2D2', descriptors.type_name)
        self.assertEqual('R2D2', descriptors.keypoints_type)
        self.assertEqual('L2', descriptors.metric_type)
        self.assertEqual(3, len(descriptors))
        self.assertIn('a/a.jpg', descriptors)


# GlobalFeatures #######################################################################################################
class TestGlobalFeatures(unittest.TestCase):
    def test_init_global_features_unknown(self):
        global_features = kapture.GlobalFeatures('BOW', float, 4, 'CSI',
                                                 ['a/a.jpg', 'b/b.jpg', 'c/c.jpg', 'c/c.jpg'])
        self.assertEqual('BOW', global_features.type_name)
        self.assertEqual('CSI', global_features.metric_type)
        self.assertEqual(3, len(global_features))
        self.assertIn('a/a.jpg', global_features)


# Points3d #############################################################################################################
class TestPoints3d(unittest.TestCase):
    def test_empty(self):
        points3d = kapture.Points3d()
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (0, kapture.Points3d.XYZ_RGB))
        self.assertRaises(IndexError, points3d.__getitem__, (0, 0))
        self.assertFalse(points3d)

    def test_from_numpy_view(self):
        np_array = np.ones((10, 6))
        points3d = np_array.view(kapture.Points3d)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (10, kapture.Points3d.XYZ_RGB))
        self.assertTrue(points3d)

    def test_from_numpy_array(self):
        data = np.array([[0] * 6])
        points3d = kapture.Points3d(data)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (1, kapture.Points3d.XYZ_RGB))
        self.assertTrue(points3d)

        data = np.array([[0] * 3])
        points3d = kapture.Points3d(data)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (1, kapture.Points3d.XYZ_ONLY))
        self.assertTrue(points3d)

        data = np.arange(12).reshape((2, 6))
        points3d = kapture.Points3d(data)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (2, kapture.Points3d.XYZ_RGB))
        self.assertTrue(points3d)

        data = np.arange(15).reshape((5, 3))
        points3d = kapture.Points3d(data)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (5, kapture.Points3d.XYZ_ONLY))
        self.assertTrue(points3d)

        data = np.arange(16).reshape((2, 8))
        points3d = kapture.Points3d(data[:, 0:6])
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (2, kapture.Points3d.XYZ_RGB))
        self.assertTrue(points3d)

        data = list()
        data.append(list(range(1, 7)))
        points3d = kapture.Points3d(data)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (1, kapture.Points3d.XYZ_RGB))
        self.assertTrue(np.all(np.isclose(points3d.as_array(), data)))

        data = np.vstack([data, data])
        points3d = kapture.Points3d(data)
        self.assertIsInstance(points3d, kapture.Points3d)
        self.assertEqual(points3d.shape, (2, kapture.Points3d.XYZ_RGB))
        self.assertTrue(np.all(np.isclose(points3d.as_array(), data)))

    def test_from_numpy_array_invalid(self):
        data = np.arange(16).reshape((8, 2))
        with self.assertRaises(ValueError):
            kapture.Points3d(data)

        data = np.arange(16)
        with self.assertRaises(ValueError):
            kapture.Points3d(data)

    def test_slices(self):
        rows = 18
        float_array_values = np.array(range(0, rows * kapture.Points3d.XYZ_RGB), dtype=np.float64)
        np_points = np.reshape(float_array_values, (rows * 3, 2))
        self.assertRaises(ValueError, kapture.Points3d, np_points)

        np_points2 = np.reshape(np_points, (rows, kapture.Points3d.XYZ_RGB))
        # construct from numpy array
        points = kapture.Points3d(np_points2)
        # construct from a points3d
        points3d = kapture.Points3d(points)
        self.assertEqual(points, points3d)
        self.assertEqual(points3d[0, 0], 0.0)
        self.assertEqual(points3d[17, 5], 107.0)

        np.testing.assert_array_equal(points3d.as_array(), np_points2)
        np.testing.assert_array_equal(points3d[0, 0:4], float_array_values[0:4])
        np.testing.assert_array_equal(points3d[0, 4:], float_array_values[4:kapture.Points3d.XYZ_RGB])
        np.testing.assert_array_equal(points3d[0, :].as_array(), float_array_values[:kapture.Points3d.XYZ_RGB])

    def test_type_checking(self):
        self.assertRaises(TypeError, kapture.Points3d, (2, 3, 4))

        points3d = kapture.Points3d()
        self.assertEqual(points3d.dtype.name, 'float64')

        data = list()
        data.append(list(range(1, 7)))
        points3d = kapture.Points3d(data)
        self.assertEqual(points3d.dtype.name, 'float64')

        rows = 18
        float_list = np.array(range(0, rows * kapture.Points3d.XYZ_RGB))
        float_array = np.reshape(float_list, (rows, kapture.Points3d.XYZ_RGB))
        points3d = kapture.Points3d(float_array)
        self.assertEqual(points3d.dtype.name, 'float64')

        # test slicing
        val1 = points3d[:, 0:4]
        val2 = points3d[:]
        val3 = points3d[:, 0]
        val4 = points3d[0, :]
        val5 = points3d[0]
        val6 = points3d[0:2, :]
        val7 = points3d[0, 0]

        self.assertNotIsInstance(val1, kapture.Points3d)
        self.assertIsInstance(val2, kapture.Points3d)
        self.assertNotIsInstance(val3, kapture.Points3d)
        self.assertIsInstance(val4, kapture.Points3d)
        self.assertIsInstance(val5, kapture.Points3d)
        self.assertIsInstance(val6, kapture.Points3d)
        self.assertIsInstance(val7, float)


# OBSERVATIONS #########################################################################################################
class TestObservations(unittest.TestCase):
    def test_init_observations(self):
        observations = kapture.Observations({
            0: {'R2D2': [('a/a.jpg', 1), ('b/b.jpg', 2)],
                'D2NET': [('a/a.jpg', 0)]},
            1: {'R2D2': [('c/c.jpg', 1), ('c/c.jpg', 2)],
                'D2NET': [('c/c.jpg', 0), ('c/c.jpg', 1)]}
        })
        self.assertEqual(2, len(observations))
        self.assertEqual(7, observations.observations_number())
        self.assertTrue(0 in observations)
        self.assertTrue((0, 'R2D2') in observations)
        self.assertEqual(len(observations[0, 'R2D2']), 2)
        self.assertTrue((0, 'D2NET') in observations)
        self.assertEqual(len(observations[0, 'D2NET']), 1)

        self.assertTrue(1 in observations)
        self.assertTrue((1, 'R2D2') in observations)
        self.assertEqual(len(observations[1, 'R2D2']), 2)
        self.assertTrue((1, 'D2NET') in observations)
        self.assertEqual(len(observations[1, 'D2NET']), 2)

        observations.add(0, 'R2D2', 'c/c.jpg', 3)
        self.assertEqual(2, len(observations))
        self.assertEqual(8, observations.observations_number())
        self.assertEqual(len(observations[0, 'R2D2']), 3)

        observations.add(2, 'R2D2', 'a/a.jpg', 3)
        self.assertEqual(3, len(observations))
        self.assertEqual(9, observations.observations_number())
        self.assertTrue(2 in observations)
        self.assertTrue((2, 'R2D2') in observations)
        self.assertTrue((2, 'D2NET') not in observations)
        self.assertEqual(len(observations[2, 'R2D2']), 1)


# MATCHES ##############################################################################################################
class TestMatches(unittest.TestCase):
    def test_init_matches(self):
        matches = kapture.Matches(set([('bb', 'aa'), ('cc', 'dd')]))
        matches.normalize()

        self.assertFalse(('bb', 'aa') in matches)
        self.assertTrue(('aa', "bb") in matches)

        self.assertFalse(('dd', 'cc') in matches)
        self.assertTrue(('cc', 'dd') in matches)

        matches.add('ee', 'ff')
        matches.add('hh', 'gg')
        matches.normalize()

        self.assertFalse(('ff', 'ee') in matches)
        self.assertTrue(('ee', "ff") in matches)

        self.assertFalse(('hh', 'gg') in matches)
        self.assertTrue(('gg', 'hh') in matches)


# KAPTURE ##############################################################################################################
class TestKapture(unittest.TestCase):
    def test_init(self):
        lidar0 = kapture.Sensor(kapture.SensorType.lidar.name, [])
        cam0 = kapture.Sensor(kapture.SensorType.camera.name, [])
        sensors = kapture.Sensors()
        sensors['cam0'] = cam0
        kapture_data = kapture.Kapture(sensors=sensors)
        self.assertEqual(sensors, kapture_data.sensors)
        self.assertEqual(sensors['cam0'], kapture_data.sensors['cam0'])
        # assign
        sensors = kapture_data.sensors
        self.assertIsInstance(sensors, kapture.Sensors)
        kapture_data.sensors = sensors
        kapture_data.sensors['lidar0'] = lidar0

    def test_as_dict(self):
        kapture_data = kapture.Kapture()
        # test empty
        members = kapture_data.as_dict()
        self.assertEqual(members, {})
        members = kapture_data.as_dict(keep_none=True)
        self.assertEqual(len(members), 18)
        self.assertTrue(all(member is None for member in members.values()))

        # test sensors only
        kapture_data.sensors = kapture.Sensors({'cam0': kapture.Sensor(kapture.SensorType.camera.name, [])})
        members = kapture_data.as_dict()
        self.assertEqual(len(members), 1)
        self.assertEqual(members, {'sensors': kapture_data.sensors})
        members = kapture_data.as_dict(keep_none=True)
        self.assertEqual(len(members), 18)
        self.assertEqual(members['sensors'], kapture_data.sensors)
        self.assertTrue(all(member is None for name, member in members.items() if name != 'sensors'))


if __name__ == '__main__':
    unittest.main()
