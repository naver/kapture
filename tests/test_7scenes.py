#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture
import kapture
from kapture.io.csv import kapture_from_dir
from kapture.algo.compare import equal_kapture
from tools.kapture_import_7scenes import import_7scenes


class TestImport7scenes(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/'))
        self.d7s_gt_folder = path.join(samples_folder, 'chess', '7scenes')
        self.kapture_gt_folder = path.join(samples_folder, 'chess', 'kapture')

    def test_import(self):
        expected_kdata = kapture_from_dir(self.kapture_gt_folder)
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_7scenes(d7scenes_path=self.d7s_gt_folder,
                           kapture_dir_path=tmpdirname,
                           force_overwrite_existing=True)

            imported_kdata = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_kdata, expected_kdata))


if __name__ == '__main__':
    unittest.main()
