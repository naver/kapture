#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.io.features import FEATURE_FILE_EXTENSION, get_keypoints_fullpath, image_keypoints_from_file
from kapture.algo.compare import equal_kapture, equal_sensors, equal_records_gnss
import kapture.io.csv as csv
import numpy as np


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

        with csv.get_all_tar_handlers(self._kapture_tar_sample_path) as tar_handlers:
            kapture_data_actual = csv.kapture_from_dir(self._kapture_tar_sample_path, tar_handlers=tar_handlers)

            # check in detail what could be wrong
            self.assertTrue(equal_sensors(kapture_data_expected.sensors, kapture_data_actual.sensors))
            self.assertTrue(equal_records_gnss(kapture_data_expected.records_gnss, kapture_data_actual.records_gnss))
            # check all at once
            self.assertTrue(equal_kapture(kapture_data_expected, kapture_data_actual))

            # check kpt
            image_name = '01.jpg'
            keypoints_type = 'HessianAffine'
            kapture_keypoints_filepath = get_keypoints_fullpath(keypoints_type,
                                                                kapture_dirpath=self._kapture_sample_path,
                                                                image_filename=image_name)
            kpts_expected = image_keypoints_from_file(kapture_keypoints_filepath,
                                                      kapture_data_expected.keypoints[keypoints_type].dtype,
                                                      kapture_data_expected.keypoints[keypoints_type].dsize)
            handler_kpts = tar_handlers.keypoints[keypoints_type]
            kpts_actual = handler_kpts.get_array_from_tar(image_name + FEATURE_FILE_EXTENSION[kapture.Keypoints],
                                                          kapture_data_expected.keypoints[keypoints_type].dtype,
                                                          kapture_data_expected.keypoints[keypoints_type].dsize)
            self.assertTrue(np.array_equal(kpts_expected, kpts_actual))

            kapture_keypoints_filepath = get_keypoints_fullpath(keypoints_type,
                                                                kapture_dirpath=self._kapture_sample_path,
                                                                image_filename=image_name,
                                                                tar_handler=tar_handlers)
            kpts_expected = image_keypoints_from_file(kapture_keypoints_filepath,
                                                      kapture_data_expected.keypoints[keypoints_type].dtype,
                                                      kapture_data_expected.keypoints[keypoints_type].dsize)
            self.assertTrue(np.array_equal(kpts_expected, kpts_actual))


if __name__ == '__main__':
    unittest.main()
