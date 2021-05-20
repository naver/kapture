#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import numpy as np
import quaternion
import copy
import os.path as path
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
import kapture.io.csv as csv
from kapture.algo.compare import is_distance_within_threshold, pose_transform_distance
from kapture.algo.compare import equal_kapture, equal_sensors, equal_rigs, equal_trajectories, equal_records_camera,\
    equal_records_lidar, equal_records_wifi, equal_records_gnss, equal_poses, equal_keypoints_collections


class TestComparePoseTransform(unittest.TestCase):
    """
    Testing kapture/algorithms/compare.py
    """

    def setUp(self):
        self.rotation_a = quaternion.quaternion(-0.572, 0.198, 0.755, -0.252)
        self.rotation_a_negative = quaternion.quaternion(0.572, -0.198, -0.755, 0.252)

        self.translation_a = [144.801, -74.548, -17.746]

        self.pose_a = kapture.PoseTransform(self.rotation_a, self.translation_a)
        self.pose_a_negative = kapture.PoseTransform(self.rotation_a_negative, self.translation_a)

        self.rotation_b = quaternion.quaternion(0.878, 0.090, 0.374, -0.285)
        self.rotation_b_negative = quaternion.quaternion(-0.878, -0.090, -0.374, 0.285)

        self.translation_b = [4.508,  45.032, -37.840]

        self.pose_b = kapture.PoseTransform(self.rotation_b, self.translation_b)
        self.pose_b_negative = kapture.PoseTransform(self.rotation_b_negative, self.translation_b)

        self.pose_ab = kapture.PoseTransform(self.rotation_a, self.translation_b)
        self.pose_ba = kapture.PoseTransform(self.rotation_b, self.translation_a)

        self.pose_none = kapture.PoseTransform(r=None, t=None)
        self.pose_r_none = kapture.PoseTransform(r=None, t=self.translation_a)
        self.pose_t_none = kapture.PoseTransform(r=self.rotation_a, t=None)

    def test_is_distance_within_threshold(self):
        self.assertTrue(is_distance_within_threshold(pose_transform_distance(self.pose_a, self.pose_a)))
        self.assertTrue(is_distance_within_threshold(pose_transform_distance(self.pose_a_negative,
                                                                             self.pose_a_negative)))
        self.assertTrue(is_distance_within_threshold(pose_transform_distance(self.pose_a, self.pose_a_negative)))
        self.assertTrue(is_distance_within_threshold(pose_transform_distance(self.pose_a,
                                                                             self.pose_a.inverse().inverse())))
        self.assertTrue(is_distance_within_threshold(pose_transform_distance(self.pose_b, self.pose_b_negative)))
        self.assertFalse(is_distance_within_threshold(pose_transform_distance(self.pose_a, self.pose_a.inverse())))
        self.assertFalse(is_distance_within_threshold(pose_transform_distance(self.pose_a_negative,
                                                                              self.pose_a_negative.inverse())))
        self.assertFalse(is_distance_within_threshold(pose_transform_distance(self.pose_a, self.pose_b)))
        self.assertFalse(is_distance_within_threshold(pose_transform_distance(self.pose_a, self.pose_ab)))
        self.assertFalse(is_distance_within_threshold(pose_transform_distance(self.pose_a, self.pose_ba)))
        self.assertFalse(is_distance_within_threshold(pose_transform_distance(self.pose_ab, self.pose_ba)))

    def test_equal_poses(self):
        self.assertTrue(equal_poses(self.pose_none, self.pose_none))
        self.assertFalse(equal_poses(self.pose_none, self.pose_r_none))
        self.assertFalse(equal_poses(self.pose_none, self.pose_t_none))
        self.assertFalse(equal_poses(self.pose_t_none, self.pose_r_none))
        self.assertFalse(equal_poses(self.pose_r_none, self.pose_t_none))
        self.assertTrue(equal_poses(self.pose_t_none, self.pose_t_none))
        self.assertTrue(equal_poses(self.pose_r_none, self.pose_r_none))
        self.assertTrue(equal_poses(self.pose_a, self.pose_a))
        self.assertFalse(equal_poses(self.pose_a, self.pose_b))
        self.assertTrue(equal_poses(self.pose_a, self.pose_a_negative))


class TestCompareGnss(unittest.TestCase):
    """
    Testing kapture/algorithms/compare.py
    """

    def setUp(self):
        self.gnss_a = kapture.RecordsGnss()
        self.gnss_b = kapture.RecordsGnss()

        # build 2 gps track using 2 different insertion methods
        # gnss_a[timestamp, gps_id] = values and gnss_b[timestamp] = {gps_id: values}
        for timestamp in range(3):
            gps_snapshots = {}
            for gps_id in ['gps1', 'gps2']:
                values = np.random.uniform(0., 50., size=(5, )).tolist()
                self.gnss_a[timestamp, gps_id] = kapture.RecordGnss(*values)
                gps_snapshots[gps_id] = kapture.RecordGnss(*values)
            self.gnss_b[timestamp] = gps_snapshots

    def test_compare_gnss_equal(self):
        # check they are effectively equals
        self.assertTrue(equal_records_gnss(self.gnss_a, self.gnss_b))

    def test_compare_gnss_modify(self):
        # then modify one and check its not equal anymore
        self.gnss_b[0, 'gps1'].x = 0
        self.assertFalse(equal_records_gnss(self.gnss_a, self.gnss_b))

    def test_compare_gnss_ablation(self):
        # then modify one and check its not equal anymore
        self.gnss_b[0].pop('gps1')
        self.assertFalse(equal_records_gnss(self.gnss_a, self.gnss_b))

    def test_compare_gnss_addition(self):
        # then modify one and check its not equal anymore
        self.gnss_b[0, 'gps3'] = self.gnss_b[0, 'gps1']
        self.assertFalse(equal_records_gnss(self.gnss_a, self.gnss_b))


class TestCompareM1x(unittest.TestCase):

    def setUp(self):
        self._samples_folder = path.abspath(path.join(path.dirname(__file__), '..', 'samples', 'm1x'))
        self._kapture_data = csv.kapture_from_dir(self._samples_folder)

        rotation_a = quaternion.quaternion(-0.572, 0.198, 0.755, -0.252)
        rotation_b = quaternion.quaternion(0.878, 0.090, 0.374, -0.285)
        translation_a = [144.801, -74.548, -17.746]
        translation_b = [144.701, -73.548, -17.746]
        self._pose_a = kapture.PoseTransform(rotation_a, translation_a)
        self._pose_b = kapture.PoseTransform(rotation_b, translation_b)

        self._rigs = kapture.Rigs()
        self._rigs['rig_a', '144323'] = self._pose_a
        self._rigs['rig_b', '144323'] = self._pose_b

        # self._rig_a = kapture.Rig()
        # self._rig_a['144323'] = self._pose_a
        # self._rig_b = kapture.Rig()
        # self._rig_b['144323'] = self._pose_b

    def test_copy_equal(self):
        kapture_data_copy = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_kapture(kapture_data_copy, self._kapture_data))

    def test_equal_sensors(self):
        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_sensors(kapture_data_a.sensors, kapture_data_b.sensors))

        kapture_data_a.sensors['7497487'] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, [3882, 382])
        self.assertFalse(equal_sensors(kapture_data_a.sensors, kapture_data_b.sensors))

        kapture_data_b.sensors['7497487'] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, [3882, 382])
        self.assertTrue(equal_sensors(kapture_data_a.sensors, kapture_data_b.sensors))

        kapture_data_b.sensors['7497487'] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, [3882, 383])
        self.assertFalse(equal_sensors(kapture_data_a.sensors, kapture_data_b.sensors))

    def test_equal_rigs(self):

        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_rigs(kapture_data_a.rigs, kapture_data_b.rigs))

        kapture_data_a.rigs['497376'] = self._rigs['rig_a']
        self.assertFalse(equal_rigs(kapture_data_a.rigs, kapture_data_b.rigs))

        kapture_data_b.rigs['497376'] = self._rigs['rig_a']
        self.assertTrue(equal_rigs(kapture_data_a.rigs, kapture_data_b.rigs))

        kapture_data_b.rigs['497376'] = self._rigs['rig_b']
        self.assertFalse(equal_rigs(kapture_data_a.rigs, kapture_data_b.rigs))

    def test_equal_trajectories(self):
        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_trajectories(kapture_data_a.trajectories, kapture_data_b.trajectories))

        kapture_data_a.trajectories[(7497487, '497376')] = self._pose_a
        self.assertFalse(equal_trajectories(kapture_data_a.trajectories, kapture_data_b.trajectories))

        kapture_data_b.trajectories[(7497487, '497376')] = self._pose_a
        self.assertTrue(equal_trajectories(kapture_data_a.trajectories, kapture_data_b.trajectories))

        kapture_data_b.trajectories[(7497487, '497376')] = self._pose_b
        self.assertFalse(equal_trajectories(kapture_data_a.trajectories, kapture_data_b.trajectories))

    def test_equal_records_camera(self):
        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_records_camera(kapture_data_a.records_camera, kapture_data_b.records_camera))

        kapture_data_a.records_camera[(7497487, '7497487')] = 'a/a0.jpg'
        self.assertFalse(equal_records_camera(kapture_data_a.records_camera, kapture_data_b.records_camera))

        kapture_data_b.records_camera[(7497487, '7497487')] = 'a/a0.jpg'
        self.assertTrue(equal_records_camera(kapture_data_a.records_camera, kapture_data_b.records_camera))

        kapture_data_b.records_camera[(7497487, '7497487')] = 'b/b0.jpg'
        self.assertFalse(equal_records_camera(kapture_data_a.records_camera, kapture_data_b.records_camera))

    def test_equal_records_lidar(self):
        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_records_lidar(kapture_data_a.records_lidar, kapture_data_b.records_lidar))

        kapture_data_a.records_lidar[(7497487, '7497487')] = 'a/a0.pct'
        self.assertFalse(equal_records_lidar(kapture_data_a.records_lidar, kapture_data_b.records_lidar))

        kapture_data_b.records_lidar[(7497487, '7497487')] = 'a/a0.pct'
        self.assertTrue(equal_records_lidar(kapture_data_a.records_lidar, kapture_data_b.records_lidar))

        kapture_data_b.records_lidar[(7497487, '7497487')] = 'b/b0.pct'
        self.assertFalse(equal_records_lidar(kapture_data_a.records_lidar, kapture_data_b.records_lidar))

    def test_equal_records_wifi(self):
        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        # check for deep copy
        self.assertTrue(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))

        record_wifi_1 = kapture.RecordWifi({
            '68:72:51:80:52:df': kapture.RecordWifiSignal(frequency=2417, rssi=-33.0, ssid='M1X_PicoM2'),
            '68:9c:e2:e1:b0:60': kapture.RecordWifiSignal(frequency=5765, rssi=-49.0, ssid='@HYUNDAI-WiFi'),
        })
        record_wifi_2 = kapture.RecordWifi({
            '68:72:51:80:52:df': kapture.RecordWifiSignal(frequency=2417, rssi=-35.0, ssid='M1X_PicoM2'),
            '68:9c:e2:e1:b0:60': kapture.RecordWifiSignal(frequency=5765, rssi=-47.0, ssid='@HYUNDAI-WiFi'),
        })
        record_wifi_2_str = kapture.RecordWifi({
            '68:72:51:80:52:df': kapture.RecordWifiSignal(frequency='2417', rssi='-35.0', ssid='M1X_PicoM2'),
            '68:9c:e2:e1:b0:60': kapture.RecordWifiSignal(frequency='5765', rssi='-47.0', ssid='@HYUNDAI-WiFi'),
        })
        # check deep copy just did nothing: recreate the B
        kapture_data_b.records_wifi = kapture.RecordsWifi()
        kapture_data_b.records_wifi[1555398770307, 'AC01324954_wifi'] = record_wifi_1
        kapture_data_b.records_wifi[1555398771307, 'AC01324954_wifi'] = record_wifi_2
        self.assertTrue(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))
        # check str arguments are casted to the proper types
        kapture_data_b.records_wifi[1555398771307, 'AC01324954_wifi'] = record_wifi_2_str
        self.assertTrue(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))

        # check if some are missing, its not equal
        kapture_data_b.records_wifi = kapture.RecordsWifi()
        kapture_data_b.records_wifi[1555398770307, 'AC01324954_wifi'] = record_wifi_1
        self.assertFalse(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))

        # just modify a single frequency
        kapture_data_b = copy.deepcopy(self._kapture_data)
        kapture_data_b.records_wifi[1555398770307, 'AC01324954_wifi']['68:72:51:80:52:df'].frequency += 1
        self.assertFalse(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))

        # just modify a single rssi
        kapture_data_b = copy.deepcopy(self._kapture_data)
        kapture_data_b.records_wifi[1555398770307, 'AC01324954_wifi']['68:72:51:80:52:df'].rssi += 2.0
        self.assertFalse(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))

        # just modify a single bssid
        kapture_data_b = copy.deepcopy(self._kapture_data)
        dropped = kapture_data_b.records_wifi[1555398770307, 'AC01324954_wifi'].pop('68:72:51:80:52:df')
        self.assertFalse(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))
        kapture_data_b.records_wifi[1555398770307, 'AC01324954_wifi']['XX:XX:XX:XX:XX:XX'] = dropped
        self.assertFalse(equal_records_wifi(kapture_data_a.records_wifi, kapture_data_b.records_wifi))

    def test_equal_keypoints(self):
        kapture_data_a = copy.deepcopy(self._kapture_data)
        kapture_data_b = copy.deepcopy(self._kapture_data)
        self.assertTrue(equal_keypoints_collections(kapture_data_a.keypoints, kapture_data_b.keypoints))

        kapture_data_a.keypoints['SIFT'].add('a/a0.pct')
        self.assertFalse(equal_keypoints_collections(kapture_data_a.keypoints, kapture_data_b.keypoints))

        kapture_data_b.keypoints['SIFT'].add('a/a0.pct')
        self.assertTrue(equal_keypoints_collections(kapture_data_a.keypoints, kapture_data_b.keypoints))

        kapture_data_b.keypoints['SIFT'].remove('a/a0.pct')
        kapture_data_b.keypoints['SIFT'].add('b/b0.jpg')
        self.assertFalse(equal_keypoints_collections(kapture_data_a.keypoints, kapture_data_b.keypoints))


if __name__ == '__main__':
    unittest.main()
