#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import shutil
import tempfile
import unittest
import sys
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.io.csv import kapture_from_dir
from kapture.io.records import TransferAction
from tools.kapture_import_image_folder import import_image_folder


class TestImageFolder(unittest.TestCase):
    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/Aachen-Day-Night'))
        self.images_folder = path.join(samples_folder, 'images_upright', 'query', 'day')

    @unittest.skipIf(sys.platform.startswith("win"), "Do not work on Windows without admin rights")
    def test_import_image_folder(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            import_image_folder(
                images_path=self.images_folder,
                kapture_path=tmpdirname,
                force_overwrite_existing=True,
                images_import_method=TransferAction.link_absolute)

            kapture_result = kapture_from_dir(tmpdirname)

        # sensors
        sensors = list(kapture.flatten(kapture_result.sensors, is_sorted=True))
        self.assertEqual(4, len(sensors))
        first_sensor = sensors[0][1]
        self.assertTrue(isinstance(first_sensor, kapture.Sensor))
        self.assertTrue(isinstance(first_sensor, kapture.Camera))
        self.assertEqual(kapture.CameraType.UNKNOWN_CAMERA, first_sensor.camera_type)
        # images
        images = list(kapture.flatten(kapture_result.records_camera, is_sorted=True))
        self.assertEqual(4, len(images))
        first_image = images[0]
        self.assertEqual((0, 'sensor0', 'milestone/2011-12-17_14-28-15_257.jpg'), first_image)

    @unittest.skipIf(sys.platform.startswith("win"), "Do not work on Windows without admin rights")
    def test_import_image_folder_with_single_sensor(self):
        # creates a folder with both images and sensors.txt
        with tempfile.TemporaryDirectory() as tmpdirname_in, tempfile.TemporaryDirectory() as tmpdirname_out:
            # copy images
            shutil.copytree(self.images_folder, tmpdirname_in, dirs_exist_ok=True)
            # copy sensors.txt
            sensors_file_path = path.join(tmpdirname_in, 'sensors.txt')
            with open(sensors_file_path, 'wt') as f:
                f.write("""
                cam0, , camera, RADIAL, 2832, 2128, 2986.7801886485645, 1463.2779641275515, 1112.2674656244137, \
                -0.2465279459531899, 0.2675790139425493
                """)

            import_image_folder(tmpdirname_in, tmpdirname_out, True, TransferAction.link_absolute)

            kapture_result = kapture_from_dir(tmpdirname_out)

        sensors = list(kapture.flatten(kapture_result.sensors, is_sorted=True))
        # sensors
        self.assertEqual(1, len(sensors))
        camera_id = sensors[0][0]
        camera = sensors[0][1]
        self.assertEqual('cam0', camera_id)
        self.assertTrue(isinstance(camera, kapture.Sensor))
        self.assertTrue(isinstance(camera, kapture.Camera))
        self.assertEqual(kapture.CameraType.RADIAL, camera.camera_type)
        # images
        images = list(kapture.flatten(kapture_result.records_camera, is_sorted=True))
        self.assertEqual(4, len(images))
        first_image = images[0]
        self.assertEqual((0, 'cam0', 'milestone/2011-12-17_14-28-15_257.jpg'), first_image)

    @unittest.skipIf(sys.platform.startswith("win"), "Do not work on Windows without admin rights")
    def test_import_image_folder_with_multiple_sensors(self):
        # creates a folder with both images and sensors.txt
        with tempfile.TemporaryDirectory() as tmpdirname_in, tempfile.TemporaryDirectory() as tmpdirname_out:
            # copy images
            shutil.copytree(self.images_folder, tmpdirname_in, dirs_exist_ok=True)
            # make up sensors.txt
            sensors_file_path = path.join(tmpdirname_in, 'sensors.txt')
            with open(sensors_file_path, 'wt') as f:
                f.write("""
                milestone, , camera, RADIAL, 2832, 2128, 2986.7801886485645, 1463.2779641275515, 1112.2674656244137, \
                -0.2465279459531899, 0.2675790139425493
                nexus4, , camera, RADIAL, 2832, 2128, 2986.7801886485645, 1463.2779641275515, 1112.2674656244137, \
                -0.2465279459531899, 0.2675790139425493
                """)

            import_image_folder(tmpdirname_in, tmpdirname_out, True, TransferAction.link_absolute)

            kapture_result = kapture_from_dir(tmpdirname_out)

        sensors = list(kapture.flatten(kapture_result.sensors, is_sorted=True))
        # sensors
        self.assertEqual(2, len(sensors))
        camera_ids = [s[0] for s in sensors]
        cameras = [s[1] for s in sensors]
        self.assertEqual(['milestone', 'nexus4'], camera_ids)
        self.assertTrue(kapture.Sensor, cameras[0])
        self.assertTrue(kapture.Sensor, cameras[1])
        self.assertTrue(kapture.Camera, cameras[0])
        self.assertTrue(kapture.Camera, cameras[1])
        self.assertEqual(kapture.CameraType.RADIAL, cameras[0].camera_type)
        self.assertEqual(kapture.CameraType.RADIAL, cameras[1].camera_type)
        # images
        images = list(kapture.flatten(kapture_result.records_camera, is_sorted=True))
        self.assertEqual(4, len(images))
        first_image = images[0]
        self.assertEqual((0, 'milestone', 'milestone/2011-12-17_14-28-15_257.jpg'), first_image)
