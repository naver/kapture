#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Test some of the utils functions and classes.
"""

import datetime
import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
import kapture.utils.computation as computation
import kapture.utils.paths


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup before every test
        """
        pass

    def testRemoveFileForce(self):
        # Creates a random file
        one_file = tempfile.NamedTemporaryFile().name
        with open(one_file, 'w') as f_dst:
            f_dst.write('some content')
        self.assertTrue(path.isfile(one_file), "Test file created")
        # Remove it
        kapture.utils.paths.safe_remove_file(one_file, True)
        self.assertFalse(path.isfile(one_file), "Test file successfully removed")

    def testPrependToFile(self):
        # Creates a random file
        one_file = tempfile.NamedTemporaryFile(suffix='.txt').name
        with open(one_file, 'w') as f_dst:
            f_dst.write('some content')
        self.assertTrue(path.isfile(one_file), "Test file created")
        additional = 'new first line\n'
        kapture.utils.paths.prepend_to_file(one_file, additional)
        with open(one_file, 'r') as f:
            first_line = f.readline()
            self.assertEqual(additional, first_line, "Successful prepend")

    def testDigitsNumNotAnInt(self):
        now = datetime.datetime.now().timestamp()
        self.assertRaises(TypeError, computation.num_digits, now, msg="must be called with an integer")

    def testDigitsZero(self):
        number = 0
        self.assertEqual(computation.num_digits(number), 1, "1 digit")

    def testDigitsNum(self):
        number = 1234567890
        self.assertEqual(computation.num_digits(number), 10, "10 digits")

    def testDigitsNumEpoch(self):
        # Computes number of digits of typical epoch in microseconds
        now = int(datetime.datetime.now().timestamp()*1000000)
        self.assertEqual(computation.num_digits(now), 16, "Epoch now has 16 digits")

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        pass


if __name__ == '__main__':
    unittest.main()
