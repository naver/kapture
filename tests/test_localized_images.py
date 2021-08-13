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
from tools.kapture_import_localized_images import import_localized_images


class TestImageList(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/cour_carree_markers/'))
        self._localized_file = path.join(samples_folder, 'map/localized_images.txt')
        self._images_folder = path.join(samples_folder, 'map')
        self._kapture_with_images_path = path.join(samples_folder, 'kapture_images')
        self._kapture_no_images_path = path.join(samples_folder, 'kapture_no_images')
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_path = path.join(self._tempdir.name, 'kapture_local')
        os.makedirs(self._kapture_path, exist_ok=True)

    def test_import_images_missing_file(self):
        self.assertRaises(FileNotFoundError, import_localized_images,
                          '', '', self._kapture_path, True, TransferAction.copy, False)

    def test_import_images_missing_dir(self):
        self.assertRaises(ValueError, import_localized_images,
                          self._localized_file, '', self._kapture_path, True, TransferAction.copy, False)

    def test_import_with_images(self):
        import_localized_images(self._localized_file, self._images_folder, self._kapture_path, True,
                                TransferAction.copy, False)

        expected_kapture = kapture_from_dir(self._kapture_with_images_path)
        imported_kapture = kapture_from_dir(self._kapture_path)
        self.assertTrue(equal_kapture(imported_kapture, expected_kapture))

    def test_import_no_images(self):
        import_localized_images(self._localized_file, '', self._kapture_path, True, TransferAction.copy, True)

        expected_kapture = kapture_from_dir(self._kapture_no_images_path)
        imported_kapture = kapture_from_dir(self._kapture_path)
        self.assertTrue(equal_kapture(imported_kapture, expected_kapture))


if __name__ == '__main__':
    unittest.main()
