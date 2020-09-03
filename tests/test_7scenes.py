#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
from kapture.io.csv import kapture_from_dir
from kapture.algo.compare import equal_kapture
from kapture.io.records import TransferAction
from tools.kapture_import_7scenes import import_7scenes


class TestImport7scenes(unittest.TestCase):

    def setUp(self):
        d7scenes_sample_dirpath = path.abspath(path.join(path.dirname(__file__),  '../samples/7scenes'))
        self.d7scenes_rootpath = path.join(d7scenes_sample_dirpath, 'microsoft', 'stairs')
        self.kapture_gt_rootpath = path.join(d7scenes_sample_dirpath, 'kapture', 'stairs')
        self._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_import_both(self):
        kapture_gt_dirpath = path.join(self.kapture_gt_rootpath, 'both')
        import_7scenes(d7scenes_path=self.d7scenes_rootpath,
                       kapture_dir_path=self._tempdir.name,
                       force_overwrite_existing=True,
                       images_import_method=TransferAction.copy)

        imported_kdata = kapture_from_dir(self._tempdir.name)
        expected_kdata = kapture_from_dir(kapture_gt_dirpath)
        self.assertTrue(equal_kapture(imported_kdata, expected_kdata))

    def test_import_mapping(self):
        kapture_gt_dirpath = path.join(self.kapture_gt_rootpath, 'mapping')
        import_7scenes(d7scenes_path=self.d7scenes_rootpath,
                       kapture_dir_path=self._tempdir.name,
                       force_overwrite_existing=True,
                       partition='mapping')

        imported_kdata = kapture_from_dir(self._tempdir.name)
        expected_kdata = kapture_from_dir(kapture_gt_dirpath)
        self.assertTrue(equal_kapture(imported_kdata, expected_kdata))

    def test_import_query(self):
        kapture_gt_dirpath = path.join(self.kapture_gt_rootpath, 'query')
        import_7scenes(d7scenes_path=self.d7scenes_rootpath,
                       kapture_dir_path=self._tempdir.name,
                       force_overwrite_existing=True,
                       partition='query')

        imported_kdata = kapture_from_dir(self._tempdir.name)
        expected_kdata = kapture_from_dir(kapture_gt_dirpath)
        self.assertTrue(equal_kapture(imported_kdata, expected_kdata))

    def test_import_sequence(self):
        kapture_gt_dirpath = path.join(self.kapture_gt_rootpath, 'seq-01')
        import_7scenes(d7scenes_path=self.d7scenes_rootpath + '/seq-01',
                       kapture_dir_path=self._tempdir.name,
                       force_overwrite_existing=True)

        imported_kdata = kapture_from_dir(self._tempdir.name)
        expected_kdata = kapture_from_dir(kapture_gt_dirpath)
        self.assertTrue(equal_kapture(imported_kdata, expected_kdata))


if __name__ == '__main__':
    unittest.main()
