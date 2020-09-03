#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.io.csv import kapture_from_dir
from kapture.algo.compare import equal_kapture
from tools.kapture_import_bundler import import_bundler


class TestImportBundler(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/'))
        self.aachen_folder = path.join(samples_folder, 'Aachen-Day-Night')
        self.aachen_models_folder = path.join(self.aachen_folder, '3D-models')
        self.images_folder = path.join(self.aachen_folder, 'images_upright')
        self.bundler_sensors = kapture.Sensors()
        self.bundler_sensors['sensor0'] = kapture.Camera(kapture.CameraType.RADIAL,
                                                         [1600, 1067, 1.084590000e+03,
                                                          800, 533.5, 0.000000000e+00, 6.894198313e-08])
        self.bundler_sensors['sensor1'] = kapture.Camera(kapture.CameraType.RADIAL,
                                                         [1200, 1600, 1.556980000e+03,
                                                          600, 800, 0.000000000e+00, 3.565154420e-08])
        self.bundler_sensors['sensor2'] = kapture.Camera(kapture.CameraType.RADIAL,
                                                         [1600, 1067, 1.103400000e+03,
                                                          800, 533.5, 0.000000000e+00, 6.527248534e-08])

    def test_import(self):
        bundler_file_path = path.join(self.aachen_models_folder, 'aachen_cvpr2018_db.out')
        bundler_imagelist_path = path.join(self.aachen_models_folder, 'aachen_cvpr2018_db.list.txt')

        with tempfile.TemporaryDirectory() as tmpdirname:
            import_bundler(bundler_file_path, bundler_imagelist_path, self.images_folder, tmpdirname,
                           ignore_trajectories=False, add_reconstruction=True,
                           force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata.sensors = self.bundler_sensors
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test without trajectories
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_bundler(bundler_file_path, bundler_imagelist_path, self.images_folder, tmpdirname,
                           ignore_trajectories=True, add_reconstruction=True,
                           force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata.sensors = self.bundler_sensors
            expected_kdata._trajectories = None
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test without points3d
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_bundler(bundler_file_path, bundler_imagelist_path, self.images_folder, tmpdirname,
                           ignore_trajectories=False, add_reconstruction=False,
                           force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata.sensors = self.bundler_sensors
            expected_kdata._points3d = None
            expected_kdata._keypoints = None
            expected_kdata._observations = None
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))


if __name__ == '__main__':
    unittest.main()
