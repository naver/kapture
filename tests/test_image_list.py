#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.io.csv import kapture_from_dir
from kapture.algo.compare import equal_kapture
from tools.kapture_import_image_list import import_image_list
from tools.kapture_export_image_list import export_image_list


class TestImageListAachen(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/'))
        self.aachen_folder = path.join(samples_folder, 'Aachen-Day-Night')
        self.aachen_models_folder = path.join(self.aachen_folder, '3D-models')
        self.images_folder = path.join(self.aachen_folder, 'images_upright')
        self.query_folder = path.join(self.aachen_folder, 'queries')
        self.kapture_query_path = path.join(self.aachen_folder, 'kapture/query')

    def test_import_with_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            queries_with_intrinsics_path = path.join(self.query_folder, 'day_time_queries_with_intrinsics.txt')

            import_image_list([queries_with_intrinsics_path], self.images_folder, tmpdirname,
                              force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(self.kapture_query_path)
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

    def test_import_without_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            queries_without_intrinsics_path = path.join(self.query_folder, 'day_time_queries_without_intrinsics.txt')

            import_image_list([queries_without_intrinsics_path], self.images_folder, tmpdirname,
                              force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(self.kapture_query_path)
            # set all sensors to unknown
            for sensor_id in expected_kdata.sensors.keys():
                sensor = expected_kdata.sensors[sensor_id]
                assert isinstance(sensor, kapture.Camera)
                expected_kdata.sensors[sensor_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA,
                                                                   sensor.camera_params[0:2])
            imported_aachen_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_aachen_data, expected_kdata))

    def test_export_with_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_filename = path.join(tmpdirname, 'image_list.txt')
            export_image_list(self.kapture_query_path, output_filename, export_camera_params=True, force=True)

            with open(output_filename) as file:
                exported_content = file.readlines()

            queries_with_intrinsics_path = path.join(self.query_folder, 'day_time_queries_with_intrinsics.txt')
            with open(queries_with_intrinsics_path) as file:
                expected_content = file.readlines()

            self.assertListEqual(exported_content, expected_content)

    def test_export_without_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            output_filename = path.join(tmpdirname, 'image_list.txt')
            export_image_list(self.kapture_query_path, output_filename, export_camera_params=False, force=True)

            with open(output_filename) as file:
                exported_content = file.readlines()

            queries_without_intrinsics_path = path.join(self.query_folder, 'day_time_queries_without_intrinsics.txt')
            with open(queries_without_intrinsics_path) as file:
                expected_content = file.readlines()

            self.assertListEqual(exported_content, expected_content)


if __name__ == '__main__':
    unittest.main()
