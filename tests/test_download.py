#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os
import tempfile
import tarfile
import os.path as path
import sys
import stat
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture.converter.downloader.archives as archives
# import tools.kapture_download_dataset as download
# from unittest.mock import patch

"""
Test dataset list integrity
Returns error if a dataset is not reachable
"""

SLOW_TESTS = os.environ.get('SLOW_TESTS', False)

#
# class TestDownload(unittest.TestCase):
#
#     def setUp(self):
#         self.dataset_dir = path.abspath(path.join(path.dirname(__file__),  '../dataset'))
#
#     def test_update(self):
#         test_args = ["downloader", "--install_path", self.dataset_dir, "update"]
#         with patch.object(sys, 'argv', test_args):
#             self.assertEqual(download.kapture_download_dataset_cli(), 0)
#
#     @unittest.skipUnless(SLOW_TESTS, "slow test")
#     def test_all_datasets(self):
#         test_args = ["downloader", "--install_path", self.dataset_dir, "list", "--full"]
#         with patch.object(sys, 'argv', test_args):
#             self.assertEqual(download.kapture_download_dataset_cli(), 0)


class TestDownloaderPermissions(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        # make up a read only file
        test_dirname = 'permission_dir'
        test_filename = path.join(test_dirname, 'permission_file.txt')
        self.test_filepath = path.join(self._tempdir.name, test_filename)
        self.test_dirpath = path.join(self._tempdir.name, test_dirname)
        os.makedirs(self.test_dirpath)
        with open(self.test_filepath, 'wt') as f:
            f.write('permission')
        os.chmod(self.test_filepath, stat.S_IRUSR)
        # tar it
        self.tar_filepath = path.join(self._tempdir.name, 'archive.tar')
        with tarfile.open(self.tar_filepath, 'w:gz') as tar:
            tar.add(self.test_filepath, arcname=test_filename)
        # clean
        os.chmod(self.test_filepath, stat.S_IWUSR)
        os.remove(self.test_filepath)
        os.rmdir(self.test_dirpath)

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_extract_permissions(self):
        archives.untar_file(archive_filepath=self.tar_filepath, install_dirpath=self._tempdir.name)
        st = os.stat(self.test_filepath)
        self.assertTrue(st.st_mode & stat.S_IRUSR)
        self.assertTrue(st.st_mode & stat.S_IWUSR)


if __name__ == '__main__':
    unittest.main()
