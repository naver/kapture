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
from tools.kapture_import_nvm import import_nvm


class TestImportNVM(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/'))
        self.aachen_folder = path.join(samples_folder, 'Aachen-Day-Night')
        self.aachen_models_folder = path.join(self.aachen_folder, '3D-models')
        self.images_folder = path.join(self.aachen_folder, 'images_upright')
        self.filter = path.join(self.aachen_folder, 'filter.txt')

    def test_import_single_model(self):
        nvm_path = path.join(self.aachen_models_folder, 'aachen_cvpr2018_db.nvm')

        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path="", ignore_trajectories=False, add_reconstruction=True,
                       force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test without trajectories
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path="", ignore_trajectories=True, add_reconstruction=True,
                       force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata._trajectories = None
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test without points3d
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path="", ignore_trajectories=False, add_reconstruction=False,
                       force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata._points3d = None
            expected_kdata._keypoints = None
            expected_kdata._observations = None
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test with filter list
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path=self.filter, ignore_trajectories=True, add_reconstruction=False,
                       force_overwrite_existing=True)

            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(len(imported_aachen_data.records_camera.key_pairs()) == 2)
            images_set = {image_path for _, _, image_path in kapture.flatten(imported_aachen_data.records_camera)}
            self.assertTrue('db/1045.jpg' in images_set)
            self.assertFalse('db/4446.jpg' in images_set)
            self.assertTrue('db/1135.jpg' in images_set)
            self.assertIsNone(imported_aachen_data.trajectories)
            self.assertIsNone(imported_aachen_data.points3d)

    def test_import_two_models(self):
        nvm_path = path.join(self.aachen_models_folder, 'aachen_two_parts.nvm')

        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path="", ignore_trajectories=False, add_reconstruction=True,
                       force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test without trajectories
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path="", ignore_trajectories=True, add_reconstruction=True,
                       force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata._trajectories = None
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test without points3d
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path="", ignore_trajectories=False, add_reconstruction=False,
                       force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.aachen_folder, 'kapture/training'))
            expected_kdata._points3d = None
            expected_kdata._keypoints = None
            expected_kdata._observations = None
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

        # test with filter list
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_nvm(nvm_path, self.images_folder, tmpdirname,
                       filter_list_path=self.filter, ignore_trajectories=True, add_reconstruction=False,
                       force_overwrite_existing=True)

            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(len(imported_aachen_data.records_camera.key_pairs()) == 2)
            images_set = {image_path for _, _, image_path in kapture.flatten(imported_aachen_data.records_camera)}
            self.assertTrue('db/1045.jpg' in images_set)
            self.assertFalse('db/4446.jpg' in images_set)
            self.assertTrue('db/1135.jpg' in images_set)
            self.assertIsNone(imported_aachen_data.trajectories)
            self.assertIsNone(imported_aachen_data.points3d)


if __name__ == '__main__':
    unittest.main()
