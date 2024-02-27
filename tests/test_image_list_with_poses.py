#!/usr/bin/env python3
# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

import os
import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
from kapture.algo.compare import equal_kapture
from kapture.io.csv import kapture_from_dir
from kapture.io.records import TransferAction
from tools.kapture_import_image_list_with_poses import import_image_list_with_poses


class TestImageList(unittest.TestCase):

    def setUp(self):
        self._samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/cour_carree_markers/'))
        self._images_list_with_poses_file = path.join(self._samples_folder, 'map/images_with_poses.txt')
        self._images_folder = path.join(self._samples_folder, 'map')
        self._camera_model_path = path.join(path.join(self._samples_folder, 'map'), 's10.yml')
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_path = path.join(self._tempdir.name, 'kapture_local')
        os.makedirs(self._kapture_path, exist_ok=True)

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_import_images_missing_file(self):
        self.assertRaises(FileNotFoundError, import_image_list_with_poses,
                          '', self._images_folder, self._camera_model_path,
                          self._kapture_path, True, TransferAction.copy)

    def test_import_images_missing_dir(self):
        self.assertRaises(ValueError, import_image_list_with_poses,
                          self._images_list_with_poses_file, '', self._camera_model_path,
                          self._kapture_path, True, TransferAction.copy)

    def test_import_with_images(self):
        import_image_list_with_poses(self._images_list_with_poses_file, self._images_folder, self._camera_model_path,
                                     self._kapture_path, True, TransferAction.copy)

        ref_kapture_path = path.join(self._samples_folder, 'kapture_images')
        expected_kapture = kapture_from_dir(ref_kapture_path)
        imported_kapture = kapture_from_dir(self._kapture_path)
        self.assertTrue(equal_kapture(imported_kapture, expected_kapture))

    def test_import_images_no_camera(self):
        import_image_list_with_poses(self._images_list_with_poses_file, self._images_folder, '',
                                     self._kapture_path, True, TransferAction.copy)

        ref_kapture_path = path.join(self._samples_folder, 'kapture_images_no_camera')
        expected_kapture = kapture_from_dir(ref_kapture_path)
        imported_kapture = kapture_from_dir(self._kapture_path)
        self.assertTrue(equal_kapture(imported_kapture, expected_kapture))

    def test_import_no_images(self):
        import_image_list_with_poses(self._images_list_with_poses_file, '', self._camera_model_path,
                                     self._kapture_path, True, TransferAction.copy, True)

        ref_kapture_path = path.join(self._samples_folder, 'kapture_no_images')
        expected_kapture = kapture_from_dir(ref_kapture_path)
        imported_kapture = kapture_from_dir(self._kapture_path)
        self.assertTrue(equal_kapture(imported_kapture, expected_kapture))

    def test_import_no_images_nor_camera(self):
        import_image_list_with_poses(self._images_list_with_poses_file, '', '',
                                     self._kapture_path, True, TransferAction.copy, True)

        ref_kapture_path = path.join(self._samples_folder, 'kapture_no_images_nor_camera')
        expected_kapture = kapture_from_dir(ref_kapture_path)
        imported_kapture = kapture_from_dir(self._kapture_path)
        self.assertTrue(equal_kapture(imported_kapture, expected_kapture))


if __name__ == '__main__':
    unittest.main()
