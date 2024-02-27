#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Tests of various rosbag to kapture converters.
"""

import os
import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.algo.compare import equal_kapture
from kapture.core.Sensors import Camera, CameraType
import kapture.io.csv as kcsv
# tools
from kapture.converter.ros_tools.import_utbm_sensor import BB2_CAMERA_IDENTIFIERS, TOPICS_BB2
from kapture.converter.ros_tools.import_utbm_sensor import import_utbm_sensors
from kapture.utils.open_cv import import_opencv_camera_calibration
try:
    import rosbag  # noqa: F401
    from kapture.converter.ros_tools.import_rosbag import RosBagImporter
    has_rosbag = True
except ModuleNotFoundError:
    has_rosbag = False


TOPIC_ODOMETRY = '/camera/odom/sample'


@unittest.skipIf(not has_rosbag, "rosbag module is missing")
class TestImportT265Rosbag(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup before every test
        """
        samples_t265_folder = path.abspath(path.join(path.dirname(__file__), '../samples/t265'))
        self.bag_file_path = path.join(samples_t265_folder, 'trimmed_locoffice.bag')
        t265_rig_kapture = path.join(samples_t265_folder, 'rigs_only_kapture')
        kapture_data = kcsv.kapture_from_dir(t265_rig_kapture)
        self.ros_sample_kapture_path = path.join(samples_t265_folder, 'ros_kapture')
        self.tempdir = tempfile.TemporaryDirectory()
        self.kapture_path = path.join(self.tempdir.name, 'from_ros')
        self.importer = RosBagImporter(rosbag_path=self.bag_file_path,
                                       rigs=kapture_data.rigs,
                                       sensors=kapture_data.sensors,
                                       kapture_path=self.kapture_path,
                                       force_overwrite_existing=True)
        self.camera_ids = sorted(list(kapture_data.cameras.keys()))
        self.first_cam_id = self.camera_ids[0]

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def testMissingTopic(self):
        with self.assertRaisesRegex(ValueError, 'Missing image topic',
                                    msg="Missing image topic detected"):
            self.importer.import_multi_camera(odometry_topic="/message/odometry",
                                              image_topics=[],
                                              camera_identifiers=[],
                                              percent=100)
        with self.assertRaisesRegex(ValueError, 'Unequal number of .*',
                                    msg="Missing camera identifier detected"):
            self.importer.import_multi_camera(odometry_topic="/message/odometry",
                                              image_topics="/camera/image/left",
                                              camera_identifiers=[],
                                              percent=100)

    def testUnknownTopic(self):
        with self.assertRaisesRegex(ValueError, 'Missing topic .* Rosbag',
                                    msg="Unknown odometry topic detected"):
            self.importer.import_multi_camera(odometry_topic="/message/odometry",
                                              image_topics="/camera/image/left",
                                              camera_identifiers=self.first_cam_id,
                                              percent=100)
        with self.assertRaisesRegex(ValueError, 'Missing image topic .* Rosbag',
                                    msg="Unknown image topic detected"):
            self.importer.import_multi_camera(odometry_topic=TOPIC_ODOMETRY,
                                              image_topics="/camera/image/left",
                                              camera_identifiers=self.first_cam_id,
                                              percent=100)

    def testInvalidCameraIdentifiers(self):
        with self.assertRaisesRegex(ValueError, 'Camera identifier left .* not defined',
                                    msg="Invalid camera identifier"):
            self.importer.import_multi_camera(odometry_topic=TOPIC_ODOMETRY,
                                              image_topics=["/camera/image/left"],
                                              camera_identifiers=["left"],
                                              percent=100)

    def test_import_t265_rosbag(self):
        # rosbag was trimmed with
        # filter LocOffice_2019-03-21-15-57-04.bag trimmed_bag.bag "t.secs <= 1553180224 and t.nsecs <= 460916281"
        self.importer.import_multi_camera(odometry_topic=TOPIC_ODOMETRY,
                                          image_topics=['/camera/fisheye1/image_raw', '/camera/fisheye2/image_raw'],
                                          camera_identifiers=self.camera_ids,
                                          save_all_positions=False,
                                          find_image_position=True)
        rig_id = list(self.importer._rigs.keys())[0]
        self.importer.save_to_kapture(rig_id)

        rosbag_kapture_data = kcsv.kapture_from_dir(self.ros_sample_kapture_path)
        imported_data = kcsv.kapture_from_dir(self.kapture_path)
        self.assertEqual(len(imported_data.trajectories), len(rosbag_kapture_data.records_camera),
                         "one pose per image pair")
        self.assertTrue(equal_kapture(rosbag_kapture_data, imported_data), "Imported kapture ok")

        images_path = kapture.io.records.get_image_fullpath(self.kapture_path)
        images_files = []
        for root, dirs, files in os.walk(images_path):
            for name in files:
                images_files.append(path.join(root, name))
        self.assertEqual(len(images_files), 6)
        # Check the files exist
        for image_file in images_files:
            self.assertTrue(path.isfile(image_file), f"Image file {image_file} exist")

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self.tempdir.cleanup()


class TestImportUtbmRosbag(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup before every test
        """
        self._samples_utbm_folder = path.abspath(path.join(path.dirname(__file__), '../samples/utbm'))
        self.tempdir = tempfile.TemporaryDirectory()
        self.kapture_path = path.join(self.tempdir.name, 'from_ros')

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self.tempdir.cleanup()

    def test_read_bb2_camera_info(self) -> None:
        """
        Test the creation of a kapture camera object from a camera info file
        """
        cam_info_file = path.join(self._samples_utbm_folder, 'bb2.yaml')
        sensors = import_utbm_sensors(cam_info_file)
        self.assertEqual(1, len(sensors), "Created one sensor")
        camera_name = list(sensors)[0]
        self.assertEqual('bb2_cam', camera_name, "Correct camera name")
        camera = sensors[camera_name]
        self.assertIsInstance(camera, Camera, "Is of type Camera")
        self.assertEqual(CameraType.OPENCV, camera.camera_type, "of type openCV")
        self.assertEqual('bb2_cam', camera.name, "Named bb2_cam")
        self.assertEqual(1024, camera.camera_params[0], "Image width")
        self.assertEqual(768, camera.camera_params[1], "Image height")

    def test_read_bb2_with_k3_camera_info(self) -> None:
        """
        Test exception thrown when a camera info file k3 parameter is not zero
        """
        cam_info_file = path.join(self._samples_utbm_folder, 'bb2_with_k3.yaml')
        camera = import_opencv_camera_calibration(cam_info_file)
        self.assertIsInstance(camera, Camera, "Is of type Camera")
        self.assertEqual(CameraType.FULL_OPENCV, camera.camera_type, "of type full openCV")
        self.assertEqual('bb2_cam', camera.name, "Named bb2_cam")
        self.assertEqual(1024, camera.camera_params[0], "Image width")
        self.assertEqual(768, camera.camera_params[1], "Image height")
        self.assertNotEqual(0.0, camera.camera_params[10], "K3 is not null")

    @unittest.skipIf(not has_rosbag, "rosbag module is missing")
    def test_utbm_images_rosbag_import(self) -> None:
        """
        Test the import of an image rosbag
        """
        # Use a small bb2 rosbag
        rosbag_path = path.join(self._samples_utbm_folder, '2018-07-13-15-17-20_1_first10_bb2.bag')
        sensors = import_utbm_sensors([path.join(self._samples_utbm_folder, 'bb2_left.yaml'),
                                       path.join(self._samples_utbm_folder, 'bb2_right.yaml')])
        importer = RosBagImporter(rosbag_path, None, sensors, self.kapture_path, force_overwrite_existing=True)
        importer.import_multi_camera(None, TOPICS_BB2, BB2_CAMERA_IDENTIFIERS, True, False, False)
        importer.save_to_kapture()
        ros_sample_kapture_path = path.join(self._samples_utbm_folder, '2018-07-13-15-17-20_1_first10_bb2_kapture')
        rosbag_kapture_data = kcsv.kapture_from_dir(ros_sample_kapture_path)
        imported_data = kcsv.kapture_from_dir(self.kapture_path)
        self.assertTrue(equal_kapture(rosbag_kapture_data, imported_data), "Imported UTBM kapture ok")

        images_path = kapture.io.records.get_image_fullpath(self.kapture_path)
        images_files = []
        for root, dirs, files in os.walk(images_path):
            for name in files:
                images_files.append(path.join(root, name))
        self.assertEqual(len(images_files), 10)
        # Check the files exist
        for image_file in images_files:
            self.assertTrue(path.isfile(image_file), f"Image file {image_file} exist")


if __name__ == '__main__':
    unittest.main()
