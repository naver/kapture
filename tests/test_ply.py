#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os.path as path
import tempfile
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
from tools.kapture_export_ply import export_ply


class TestPlot(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__), '../samples/'))
        self.toplot_folder = path.join(samples_folder, 'maupertuis', 'kapture')

    def test_plot(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            export_ply(
                kapture_path=self.toplot_folder,
                ply_dir_path=tmpdirname,
                axis_length=2.0,
                only=[], skip=[])
            self.assertTrue(path.isfile(path.join(tmpdirname, "points3d.ply")))
            self.assertTrue(path.isfile(path.join(tmpdirname, "trajectories.ply")))


if __name__ == '__main__':
    unittest.main()
