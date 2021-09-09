#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os
import os.path as path
import sys
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import tools.kapture_download_dataset as download
from unittest.mock import patch

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
#

if __name__ == '__main__':
    unittest.main()
