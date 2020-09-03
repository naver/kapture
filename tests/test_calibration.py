#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import numpy as np
import quaternion
import path_to_kapture  # enables import kapture  # noqa: F401
from kapture import Rigs, Trajectories, PoseTransform, flatten
from kapture.algo.calibration import rigs_calibrate_average
from kapture.algo.pose_operations import average_quaternion


class TestRotationAvg(unittest.TestCase):
    def test_avg(self):
        # distribute a bunch of rotation around I,
        # such that the average should be I:
        # at random axis but constant angle,
        nb_rotations = 1000
        angles = np.linspace(0.0, np.deg2rad(45), 100)
        averages_q = []
        np.random.seed(1)
        for angle in angles:
            rotation_axis = np.random.uniform(-1., 1., size=(nb_rotations, 3))
            rotation_axis /= np.linalg.norm(rotation_axis, axis=1, keepdims=True)
            rotation_axis *= angle
            rotation_qs = quaternion.from_rotation_vector(rotation_axis)
            rotation_qs = quaternion.as_float_array(rotation_qs)
            # print(rotation_qs)
            r = average_quaternion(rotation_qs)
            averages_q.append(r.tolist())
        averages_q = np.array(averages_q)
        # normalizes rotation (resolve the opposite ambiguity)
        averages_q *= np.sign(averages_q[:, 0]).reshape((-1, 1))
        # convert quaternions to angle-axis representation
        averages_q = quaternion.from_float_array(averages_q)
        averages_v = quaternion.as_rotation_vector(averages_q)
        # retrieve angle amplitudes.
        errors_rad = np.linalg.norm(averages_v, axis=1)
        # # plot for debug
        # import matplotlib.pyplot as plt
        # plt.plot(np.rad2deg(errors_rad), 'x', label='Errors (in deg.) as a function of angle amplitude (in deg.)')
        # plt.show()
        maximum_error_deg = np.rad2deg(np.max(errors_rad))
        self.assertLess(maximum_error_deg, 3.6)


class TestCalibrate(unittest.TestCase):
    def setUp(self) -> None:
        self.rigs_expected = Rigs()
        # kind of make a rosace around a central point
        nb_cams = 6
        rig_id = 'rig'
        rig_radius = 10
        camera_yaw_angles = np.linspace(0, 2*np.pi, nb_cams+1)[:-1]
        t = [0, 0, -rig_radius]
        for idx, cam_yaw_angle in enumerate(camera_yaw_angles):
            r = quaternion.from_rotation_vector([0, cam_yaw_angle, 0])
            pose_cam_from_rig = PoseTransform(t=t, r=r)
            self.rigs_expected[rig_id, f'cam{idx:02d}'] = pose_cam_from_rig

        # and for the trajectory, do random
        self.rigs_trajectories = Trajectories()
        nb_timestamps = 100
        timestamps = np.arange(0, nb_timestamps)
        orientations = np.random.uniform(-1, 1., size=(nb_timestamps, 4))
        positions = np.random.uniform(-100., 100., size=(nb_timestamps, 3))
        for ts, r, t in zip(timestamps, orientations, positions):
            pose_rig_from_world = PoseTransform(t=t, r=r).inverse()
            self.rigs_trajectories[int(ts), rig_id] = pose_rig_from_world

    def test_perfect(self):
        # create the trajectory with perfect transforms
        cameras_trajectories = Trajectories()
        for timestamp, rig_id, pose_rig_from_world in flatten(self.rigs_trajectories):
            for rig_id, cam_id, pose_cam_from_rig in flatten(self.rigs_expected):
                assert rig_id == 'rig'
                pose_cam_from_world = PoseTransform.compose([pose_cam_from_rig, pose_rig_from_world])
                cameras_trajectories[timestamp, cam_id] = pose_cam_from_world

        rigs_unknown_geometry = Rigs()
        for rig_id, cam_id in self.rigs_expected.key_pairs():
            rigs_unknown_geometry[rig_id, cam_id] = PoseTransform()

        # calibrate
        rigs_recovered = rigs_calibrate_average(cameras_trajectories, rigs_unknown_geometry)

        # validate

        # debug = []
        # debug.append(pose_cam_from_world.inverse().t_raw)
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # debug = np.array(debug).transpose()
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(*debug.tolist())
        # plt.show()
        # print(debug)


if __name__ == '__main__':
    unittest.main()
