#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
from kapture.algo.compare import equal_kapture, equal_sensors, equal_records_gnss
import kapture.io.csv as csv


class TestTar(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup variables for all the tests
        """
        test_file_path = path.dirname(__file__)
        cls._kapture_sample_path = path.abspath(path.join(test_file_path, '../samples/berlin/kapture'))
        cls._kapture_tar_sample_path = path.abspath(path.join(test_file_path, '../samples/berlin/kapture_tar'))

    def test_kapture_from_dir(self) -> None:
        """
        Test the kapture_from_dir function on small sample
        """
        # convert and then load
        kapture_data_expected = csv.kapture_from_dir(self._kapture_sample_path)
        kapture_data_no_recon = csv.kapture_from_dir(self._kapture_tar_sample_path)
        # check in detail what could be wrong
        self.assertTrue(equal_sensors(kapture_data_expected.sensors, kapture_data_no_recon.sensors))
        self.assertTrue(equal_records_gnss(kapture_data_expected.records_gnss, kapture_data_no_recon.records_gnss))
        self.assertTrue(kapture_data_no_recon.keypoints is not None
                        and 'HessianAffine' in kapture_data_no_recon.keypoints
                        and len(kapture_data_no_recon.keypoints['HessianAffine']) == 0)
        self.assertTrue(kapture_data_no_recon.descriptors is not None
                        and 'HOG' in kapture_data_no_recon.descriptors
                        and len(kapture_data_no_recon.descriptors['HOG']) == 0)
        self.assertTrue(kapture_data_no_recon.matches is not None
                        and 'HessianAffine' in kapture_data_no_recon.matches
                        and len(kapture_data_no_recon.matches['HessianAffine']) == 0)
        # check all at once
        self.assertFalse(equal_kapture(kapture_data_expected, kapture_data_no_recon))

        tar_handlers = csv.get_all_tar_handlers(self._kapture_tar_sample_path)
        kapture_data_actual = csv.kapture_from_dir(self._kapture_tar_sample_path, tar_handlers=tar_handlers)

        # check in detail what could be wrong
        self.assertTrue(equal_sensors(kapture_data_expected.sensors, kapture_data_actual.sensors))
        self.assertTrue(equal_records_gnss(kapture_data_expected.records_gnss, kapture_data_actual.records_gnss))
        # check all at once
        self.assertTrue(equal_kapture(kapture_data_expected, kapture_data_actual))


if __name__ == '__main__':
    unittest.main()
