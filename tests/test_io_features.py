#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os.path as path
import numpy as np
import tempfile
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
import kapture.io.features as binary


class TestKeypoints(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._temp_filepath = path.join(self._tempdir.name, '000.jpg.kpt')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_write_uint8(self):
        nb_keypoints = 5
        dim_keypoint = 4
        the_answer = 42
        type_keypoint = np.uint8
        image_keypoints = (np.ones((nb_keypoints, dim_keypoint)) * the_answer).astype(type_keypoint)
        binary.array_to_file(self._temp_filepath, image_keypoints)
        with open(self._temp_filepath, 'rt') as file:
            line = file.readline()
        self.assertEqual('********************', line)

    def test_write_read(self):
        nb_keypoints = 5
        dim_keypoint = 4
        type_keypoint = float
        image_keypoints = np.random.random((nb_keypoints, dim_keypoint)).astype(type_keypoint)
        binary.image_keypoints_to_file(self._temp_filepath, image_keypoints)
        image_keypoints_read = binary.image_keypoints_from_file(self._temp_filepath,
                                                                dtype=type_keypoint,
                                                                dsize=dim_keypoint)
        self.assertEqual(image_keypoints.shape, image_keypoints_read.shape)
        self.assertEqual(image_keypoints.dtype, image_keypoints_read.dtype)
        self.assertAlmostEqual(image_keypoints.tolist(), image_keypoints_read.tolist())

    def test_feature_type_none(self):
        self.assertRaises(AssertionError, binary.get_features_fullpath, kapture.Keypoints, None, '')


class TestMatches(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._samples_dirpath = path.abspath(path.join(path.dirname(__file__), '..', 'samples', 'maupertuis'))
        self._kapture_dirpath = path.join(self._samples_dirpath, 'kapture')
        # self._matches_dirpath = path.join(self._kapture_dirpath, 'reconstruction', 'matches')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_matches_from_dir(self):
        self._samples_dirpath
        image_pairs_expected = {('00.jpg', '01.jpg'), ('00.jpg', '02.jpg'),
                                ('00.jpg', '03.jpg'), ('01.jpg', '02.jpg'),
                                ('01.jpg', '03.jpg'), ('02.jpg', '03.jpg')}
        image_pairs_actual = set(kapture.io.features.matching_pairs_from_dirpath('SIFT', self._kapture_dirpath))
        self.assertEqual(6, len(image_pairs_actual))
        self.assertEqual(image_pairs_expected, image_pairs_actual)

        # test matches constructor
        matches = kapture.Matches(image_pairs_expected)
        self.assertEqual(6, len(matches))
        self.assertEqual(image_pairs_expected, matches)

    def test_matches_from_file(self):
        # check a single sample file
        matching_sample_filepath = path.join(self._kapture_dirpath, 'reconstruction', 'matches', 'SIFT',
                                             '00.jpg.overlapping', '01.jpg.matches')
        matchings = kapture.io.features.image_matches_from_file(matching_sample_filepath)
        self.assertIsInstance(matchings, np.ndarray)
        self.assertEqual((2501, 3), matchings.shape)
        self.assertEqual(np.float64, matchings.dtype)
        self.assertAlmostEqual([3., 1., 0.], matchings[0].tolist())

    def test_matches_to_file(self):
        # check a single sample file
        matchings_sample_filepath = path.join(self._tempdir.name, 'dummy.matches')
        matchings = np.random.uniform(0, 10, (100, 3)).astype(np.float64)
        kapture.io.features.image_matches_to_file(matchings_sample_filepath, matchings)
        matchings_retrieved = kapture.io.features.image_matches_from_file(matchings_sample_filepath)
        self.assertAlmostEqual(matchings.tolist(), matchings_retrieved.tolist())


if __name__ == '__main__':
    unittest.main()
