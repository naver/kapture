#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import logging
import sys
import os
import os.path as path
import tempfile
import unittest
# kapture
from kapture.algo.compare import equal_kapture, equal_sensors, equal_records_gnss
import kapture.io.csv as csv
from kapture.io.records import TransferAction
import kapture
from kapture.converter.opensfm.import_opensfm import import_opensfm, import_camera
from kapture.converter.opensfm.export_opensfm import export_opensfm_camera

logger = logging.getLogger('opensfm')


class TestOpenSfM(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup variables for all the tests
        """
        logger.setLevel(logging.CRITICAL)
        cls.isWindows = sys.platform.startswith("win") or sys.platform.startswith("cygwin")
        test_file_path = path.dirname(__file__)
        cls._opensfm_sample_path = path.abspath(path.join(test_file_path, '../samples/berlin/opensfm'))
        cls._kapture_sample_path = path.abspath(path.join(test_file_path, '../samples/berlin/kapture'))

    def setUp(self) -> None:
        """
        Setup before every test
        """
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_rebuilt_path = path.join(self._tempdir.name, 'kapture_rebuilt')
        os.makedirs(self._kapture_rebuilt_path, exist_ok=True)

    def test_import_opensfm(self) -> None:
        """
        Test the import_opensfm function on small sample
        """
        # convert and then load
        file_operation = TransferAction.skip if self.isWindows else TransferAction.link_relative
        import_opensfm(
            self._opensfm_sample_path,
            self._kapture_rebuilt_path,
            force_overwrite_existing=False,
            images_import_method=file_operation
        )
        kapture_data_expected = csv.kapture_from_dir(self._kapture_sample_path)
        kapture_data_actual = csv.kapture_from_dir(self._kapture_rebuilt_path)
        # check in detail what could be wrong
        self.assertTrue(equal_sensors(kapture_data_expected.sensors, kapture_data_actual.sensors))
        self.assertTrue(equal_records_gnss(kapture_data_expected.records_gnss, kapture_data_actual.records_gnss))
        # check all at once
        self.assertTrue(equal_kapture(kapture_data_expected, kapture_data_actual))

    def test_export_opensfm(self) -> None:
        """
        Test the import_openmvg function on a small JSON file while linking the images
        """
        pass

    def test_camera_models(self) -> None:
        pano = kapture.Camera(kapture.CameraType.EQUIRECTANGULAR, [4096, 2048])
        self.assertEqual(export_opensfm_camera(pano),
                         {'projection_type': 'spherical', 'width': 4096, 'height': 2048})
        for projection_type in ['spherical', 'equirectangular']:
            imported = import_camera({'projection_type': projection_type, 'width': 4096, 'height': 2048}, 'pano')
            self.assertEqual(imported.camera_type, kapture.CameraType.EQUIRECTANGULAR)
            self.assertEqual(imported.camera_params, [4096, 2048])
        omni = kapture.Camera(kapture.CameraType.OPENCV_OMNI, [1920, 1080, 500, 500, 960, 540, 0, 1.2, 0, 0, 0, 0])
        with self.assertRaises(ValueError) as context:
            export_opensfm_camera(omni)
        self.assertIn('OPENCV_OMNI', str(context.exception))
        with self.assertRaises(ValueError):
            import_camera({'projection_type': 'dual', 'width': 10, 'height': 10})

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self._tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
