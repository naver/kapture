#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the GLIM dump import.
"""

import os
import os.path as path
import tempfile
import unittest

import numpy as np

import kapture
import kapture.io.csv as kcsv
from kapture.converter.glim.import_glim import read_submap, glim_points3d, import_glim, list_submaps


def write_submap(dump_dirpath: str, index: int, world_from_origin: np.ndarray, points: np.ndarray,
                 intensities: np.ndarray = None) -> None:
    submap_dirpath = path.join(dump_dirpath, f'{index:06d}')
    os.makedirs(submap_dirpath, exist_ok=True)
    with open(path.join(submap_dirpath, 'data.txt'), 'w') as f:
        f.write(f'id: {index}\nT_world_origin: \n')
        for row in world_from_origin:
            f.write(' '.join(f'{v:.6f}' for v in row) + '\n')
        f.write('T_origin_endpoint_L: \n' + '\n'.join(['1 0 0 0', '0 1 0 0', '0 0 1 0', '0 0 0 1']) + '\n')
        f.write('num_frames: 1\n')
    points.astype(np.float32).tofile(path.join(submap_dirpath, 'points_compact.bin'))
    if intensities is not None:
        intensities.astype(np.float32).tofile(path.join(submap_dirpath, 'intensities_compact.bin'))


class TestImportGlim(unittest.TestCase):

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.dump_dirpath = path.join(self._tempdir.name, 'dump')
        os.makedirs(self.dump_dirpath)
        # submap 0: identity; submap 1: 90 deg about z, translated by (10, 0, 1)
        self.pose0 = np.eye(4)
        self.pose1 = np.array([[0., -1., 0., 10.], [1., 0., 0., 0.], [0., 0., 1., 1.], [0., 0., 0., 1.]])
        self.points0 = np.array([[1., 0., 0.], [0., 2., 0.]])
        self.points1 = np.array([[1., 0., 0.], [0., 0., 3.]])
        write_submap(self.dump_dirpath, 0, self.pose0, self.points0, np.array([0., 100.]))
        write_submap(self.dump_dirpath, 1, self.pose1, self.points1, np.array([50., 200.]))
        with open(path.join(self.dump_dirpath, 'graph.txt'), 'w') as f:
            f.write('num_submaps: 2\n')
        with open(path.join(self.dump_dirpath, 'traj_lidar.txt'), 'w') as f:
            f.write('1791356284.469390 10 0 1 0 0 0.7071067811865476 0.7071067811865476\n')
        # kapture with the sensor and a rig
        self.kapture_dirpath = path.join(self._tempdir.name, 'kapture')
        sensors = kapture.Sensors()
        sensors['lidar0'] = kapture.Sensor(kapture.SensorType.lidar.name)
        rigs = kapture.Rigs()
        rigs['rig', 'lidar0'] = kapture.PoseTransform(r=[1., 0., 0., 0.], t=[0., 0., -0.5])
        kcsv.kapture_to_dir(self.kapture_dirpath, kapture.Kapture(sensors=sensors, rigs=rigs))

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_read_submap(self):
        submaps = list_submaps(self.dump_dirpath)
        self.assertEqual([path.basename(s) for s in submaps], ['000000', '000001'])
        world_from_origin, points, intensities = read_submap(submaps[1])
        np.testing.assert_allclose(world_from_origin, self.pose1, atol=1e-6)
        np.testing.assert_allclose(points, self.points1)
        np.testing.assert_allclose(intensities, [50., 200.])

    def test_points3d(self):
        points3d = glim_points3d(self.dump_dirpath)
        self.assertEqual(points3d.shape, (4, 3))
        expected = np.array([[1., 0., 0.], [0., 2., 0.], [10., 1., 1.], [10., 0., 4.]])
        np.testing.assert_allclose(np.asarray(points3d), expected, atol=1e-6)
        colored = glim_points3d(self.dump_dirpath, intensity_gray=True)
        self.assertEqual(colored.shape, (4, 6))
        np.testing.assert_allclose(np.asarray(colored)[:, :3], expected, atol=1e-6)
        gray = np.asarray(colored)[:, 3]
        self.assertTrue((gray >= 0).all() and (gray <= 255).all())
        self.assertLess(gray[0], gray[3])  # intensity 0 darker than 200
        np.testing.assert_array_equal(np.asarray(colored)[:, 3], np.asarray(colored)[:, 5])

    def test_import(self):
        points3d = import_glim(self.dump_dirpath, self.kapture_dirpath, intensity_gray=True,
                               trajectory_filename='traj_lidar.txt', device_id='lidar0', rig_id='rig')
        kapture_data = kcsv.kapture_from_dir(self.kapture_dirpath)
        self.assertEqual(kapture_data.points3d.shape, (4, 6))
        np.testing.assert_allclose(np.asarray(kapture_data.points3d), np.asarray(points3d), atol=1e-6)
        # trajectory: lidar at (10, 0, 1); rigs hold rig -> lidar = (0, 0, -0.5), so the rig origin is 0.5 m below
        self.assertEqual(list(kapture_data.trajectories.key_pairs()), [(1791356284469390, 'rig')])
        rig_from_world = kapture_data.trajectories[1791356284469390, 'rig']
        rig_origin = rig_from_world.inverse().t.ravel()
        np.testing.assert_allclose(rig_origin, [10., 0., 0.5], atol=1e-9)
        with self.assertRaises(FileExistsError):
            import_glim(self.dump_dirpath, self.kapture_dirpath)
        import_glim(self.dump_dirpath, self.kapture_dirpath, force_overwrite_existing=True)
        self.assertEqual(kcsv.kapture_from_dir(self.kapture_dirpath).points3d.shape, (4, 3))
        with self.assertRaises(ValueError):
            import_glim(self.dump_dirpath, self.kapture_dirpath, trajectory_filename='traj_lidar.txt',
                        force_overwrite_existing=True)  # no device_id


if __name__ == '__main__':
    unittest.main()
