#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os.path as path
import tempfile
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
from tools.kapture_export_ply import plot_ply


class TestPlot(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/'))
        self.toplot_folder = path.join(samples_folder, 'maupertuis', 'kapture')

    def test_plot(self):
        with tempfile.TemporaryDirectory() as tmpdirname:

            plot_ply(self.toplot_folder, tmpdirname, 0.2, [], [])
            self.assertTrue(path.isfile(path.join(tmpdirname, "points3d.ply"))
                            and path.isfile(path.join(tmpdirname, "trajectories.ply"))
                            and path.isfile(path.join(tmpdirname, "reconstruction", "keypoints", "00.jpg.kpt.jpg")))


if __name__ == '__main__':
    unittest.main()
