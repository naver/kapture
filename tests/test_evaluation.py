#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import numpy as np
import quaternion
import math
import copy
# kapture
import path_to_kapture  # enables import kapture
import kapture
from kapture.algo.evaluation import evaluate, evaluate_error_absolute, fill_bins


class TestEvaluation(unittest.TestCase):
    def test_error(self):
        translation = [144.801, -74, -20]
        rotation = quaternion.from_euler_angles(np.deg2rad(110.0), 0, 0)

        translation_gt = [144, -74, -20]
        rotation_gt = quaternion.from_euler_angles(np.deg2rad(125.0), 0, 0)

        pose_a = kapture.PoseTransform(r=rotation, t=translation).inverse()
        pose_a_gt = kapture.PoseTransform(r=rotation_gt, t=translation_gt).inverse()

        pose_b = kapture.PoseTransform(r=None, t=[-x for x in translation])
        pose_b_gt = kapture.PoseTransform(r=None, t=[-x for x in translation_gt])

        pose_c = kapture.PoseTransform(r=rotation.inverse(), t=None)
        pose_c_gt = kapture.PoseTransform(r=rotation_gt.inverse(), t=None)

        pose_d = kapture.PoseTransform(r=None, t=None)
        pose_d_gt = kapture.PoseTransform(r=None, t=None)

        poses = [('a', pose_a), ('b', pose_b), ('c', pose_c), ('d', pose_d), ('e', pose_d)]
        poses_gt = [('a', pose_a_gt), ('b', pose_b_gt), ('c', pose_c_gt), ('d', pose_d_gt), ('e', pose_a_gt)]
        error = evaluate_error_absolute(poses, poses_gt)

        self.assertEqual(error[0][0], 'a')
        self.assertAlmostEqual(error[0][1], 0.801)
        self.assertAlmostEqual(error[0][2], 15.0)

        self.assertEqual(error[1][0], 'b')
        self.assertAlmostEqual(error[1][1], 0.801)
        self.assertTrue(math.isnan(error[1][2]))

        self.assertEqual(error[2][0], 'c')
        self.assertTrue(math.isnan(error[2][1]))
        self.assertAlmostEqual(error[2][2], 15.0)

        self.assertEqual(error[3][0], 'd')
        self.assertTrue(math.isnan(error[3][1]))
        self.assertTrue(math.isnan(error[3][2]))

        self.assertEqual(error[4][0], 'e')
        self.assertTrue(math.isnan(error[4][1]))
        self.assertTrue(math.isnan(error[4][2]))

    def test_evaluation(self):
        position = [1.658, 0, 0]
        position_a = [2.658, 0, 0]
        position_b = [1.758, 0, 0]
        position_c = [10.1, 0, 0]
        position_d = [2., 0, 0]
        position_e = [6.658, 0, 0]

        rotation = quaternion.from_euler_angles(np.deg2rad(110.0), 0, 0)
        rotation_a = quaternion.from_euler_angles(np.deg2rad(111.0), 0, 0)
        rotation_b = quaternion.from_euler_angles(np.deg2rad(108.0), 0, 0)
        rotation_c = quaternion.from_euler_angles(np.deg2rad(10.0), 0, 0)
        rotation_d = quaternion.from_euler_angles(np.deg2rad(110.0), 0, 0)

        pose_gt = kapture.PoseTransform(r=rotation, t=position).inverse()
        pose_a = kapture.PoseTransform(r=rotation_a, t=position_a).inverse()
        pose_b = kapture.PoseTransform(r=rotation_b, t=position_b).inverse()
        pose_c = kapture.PoseTransform(r=rotation_c, t=position_c).inverse()
        pose_d = kapture.PoseTransform(r=rotation_d, t=position_d).inverse()
        pose_e = kapture.PoseTransform(r=None, t=[-x for x in position_e])

        kdata = kapture.Kapture(sensors=kapture.Sensors(),
                                records_camera=kapture.RecordsCamera(),
                                trajectories=kapture.Trajectories())
        kdata.sensors['cam0'] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, [25, 13])
        kdata.records_camera[(0, 'cam0')] = 'a'
        kdata.records_camera[(1, 'cam0')] = 'b'
        kdata.records_camera[(2, 'cam0')] = 'c'
        kdata.records_camera[(3, 'cam0')] = 'd'
        kdata.records_camera[(4, 'cam0')] = 'e'

        kdata.trajectories[(0, 'cam0')] = pose_a
        kdata.trajectories[(1, 'cam0')] = pose_b
        kdata.trajectories[(2, 'cam0')] = pose_c
        kdata.trajectories[(3, 'cam0')] = pose_d

        kdata2 = copy.deepcopy(kdata)
        kdata2.trajectories[(4, 'cam0')] = pose_e
        kdata2.records_camera[(5, 'cam0')] = 'f'

        kdata_gt = copy.deepcopy(kdata2)
        kdata_gt.trajectories[(0, 'cam0')] = pose_gt
        kdata_gt.trajectories[(1, 'cam0')] = pose_gt
        kdata_gt.trajectories[(2, 'cam0')] = pose_gt
        kdata_gt.trajectories[(3, 'cam0')] = pose_gt
        kdata_gt.trajectories[(4, 'cam0')] = pose_gt
        kdata_gt.trajectories[(5, 'cam0')] = pose_gt

        intersection = {'a', 'b', 'c', 'd', 'e'}

        result1 = evaluate(kdata, kdata_gt, intersection)
        self.assertEqual(len(result1), 5)
        self.assertEqual(result1[0][0], 'a')
        self.assertAlmostEqual(result1[0][1], 1.0)
        self.assertAlmostEqual(result1[0][2], 1.0)
        self.assertEqual(result1[1][0], 'b')
        self.assertAlmostEqual(result1[1][1], 0.1)
        self.assertAlmostEqual(result1[1][2], 2.0)
        self.assertEqual(result1[2][0], 'c')
        self.assertAlmostEqual(result1[2][1], 8.442)
        self.assertAlmostEqual(result1[2][2], 100.0)
        self.assertEqual(result1[3][0], 'd')
        self.assertAlmostEqual(result1[3][1], 0.342)
        self.assertAlmostEqual(result1[3][2], 0.0)
        self.assertEqual(result1[4][0], 'e')
        self.assertTrue(math.isnan(result1[4][1]))
        self.assertTrue(math.isnan(result1[4][2]))

        result2 = evaluate(kdata2, kdata_gt, intersection)
        self.assertEqual(len(result2), 5)
        self.assertEqual(result2[0][0], 'a')
        self.assertAlmostEqual(result2[0][1], 1.0)
        self.assertAlmostEqual(result2[0][2], 1.0)
        self.assertEqual(result2[1][0], 'b')
        self.assertAlmostEqual(result2[1][1], 0.1)
        self.assertAlmostEqual(result2[1][2], 2.0)
        self.assertEqual(result2[2][0], 'c')
        self.assertAlmostEqual(result2[2][1], 8.442)
        self.assertAlmostEqual(result2[2][2], 100.0)
        self.assertEqual(result2[3][0], 'd')
        self.assertAlmostEqual(result2[3][1], 0.342)
        self.assertAlmostEqual(result2[3][2], 0.0)
        self.assertEqual(result2[4][0], 'e')
        self.assertAlmostEqual(result2[4][1], 5.0)
        self.assertTrue(math.isnan(result2[4][2]))

        bins1 = fill_bins(result1, [(0.9, 5), (10, 105)])
        self.assertEqual(len(bins1), 2)
        self.assertEqual(bins1[0][0], 0.9)
        self.assertEqual(bins1[0][1], 5)
        self.assertEqual(bins1[0][2], 2)
        self.assertEqual(bins1[1][0], 10)
        self.assertEqual(bins1[1][1], 105)
        self.assertEqual(bins1[1][2], 4)

        bins2 = fill_bins(result1, [(0.9, 5), (10, 105)])
        self.assertEqual(len(bins2), 2)
        self.assertEqual(bins2[0][0], 0.9)
        self.assertEqual(bins2[0][1], 5)
        self.assertEqual(bins2[0][2], 2)
        self.assertEqual(bins2[1][0], 10)
        self.assertEqual(bins2[1][1], 105)
        self.assertEqual(bins2[1][2], 4)

        bins3 = fill_bins(result2, [(0.9, math.nan), (10, math.nan)])
        self.assertEqual(len(bins3), 2)
        self.assertEqual(bins3[0][0], 0.9)
        self.assertTrue(math.isnan(bins3[0][1]))
        self.assertEqual(bins3[0][2], 2)
        self.assertEqual(bins3[1][0], 10)
        self.assertTrue(math.isnan(bins3[1][1]))
        self.assertEqual(bins3[1][2], 5)

        bins4 = fill_bins(result2, [(0.9, -1), (10, -1)])
        self.assertEqual(len(bins4), 2)
        self.assertEqual(bins4[0][0], 0.9)
        self.assertEqual(bins4[0][1], -1)
        self.assertEqual(bins4[0][2], 2)
        self.assertEqual(bins4[1][0], 10)
        self.assertEqual(bins4[1][1], -1)
        self.assertEqual(bins4[1][2], 5)


if __name__ == '__main__':
    unittest.main()
