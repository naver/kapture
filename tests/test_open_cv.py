#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import unittest

from kapture.core.Sensors import CameraType
from kapture.utils.open_cv import import_opencv_camera_calibration

PINHOLE_YAML = """%YAML:1.0
image_width: 1024
image_height: 768
camera_name: bb2_left
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [831.56, 0., 510.61, 0., 833.67, 391.52, 0., 0., 1.]
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [-0.357402, 0.143604, 0.000354, 0.000730, 0.000000]
"""

OMNI_YAML = """%YAML:1.0
image_width: 1920
image_height: 1080
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [500.5, 0.25, 960., 0., 501.5, 540., 0., 0., 1.]
xi: 1.2
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 4
   dt: d
   data: [-0.1, 0.01, 0.001, -0.002]
"""

OMNI_YAML_XI_MATRIX = OMNI_YAML.replace('xi: 1.2',
                                        'xi: !!opencv-matrix\n   rows: 1\n   cols: 1\n   dt: d\n   data: [1.2]')


class TestOpenCvCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def _write(self, name: str, content: str) -> str:
        filepath = path.join(self._tempdir.name, name)
        with open(filepath, 'wt') as f:
            f.write(content)
        return filepath

    def test_pinhole(self):
        camera = import_opencv_camera_calibration(self._write('bb2.yaml', PINHOLE_YAML))
        self.assertEqual(camera.camera_type, CameraType.OPENCV)
        self.assertEqual(camera.name, 'bb2_left')
        self.assertEqual(camera.camera_params, [1024, 768, 831.56, 833.67, 510.61, 391.52,
                                                -0.357402, 0.143604, 0.000354, 0.000730])

    def test_omni(self):
        for content in [OMNI_YAML, OMNI_YAML_XI_MATRIX]:
            camera = import_opencv_camera_calibration(self._write('omni.yaml', content))
            self.assertEqual(camera.camera_type, CameraType.OPENCV_OMNI)
            self.assertEqual(camera.name, 'omni')  # no camera_name: file name
            # w, h, fx, fy, cx, cy, s, xi, k1, k2, p1, p2
            self.assertEqual(camera.camera_params, [1920, 1080, 500.5, 501.5, 960, 540, 0.25, 1.2,
                                                    -0.1, 0.01, 0.001, -0.002])
            self.assertEqual(camera.camera_param('xi'), 1.2)


if __name__ == '__main__':
    unittest.main()
