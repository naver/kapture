#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
OpenMVG import and export to kapture unit tests.
"""

import json
import logging
import numpy
import os
import os.path as path
import sys
import tempfile
from typing import Dict, List
import unittest
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.algo.compare import equal_poses, equal_kapture
import kapture.io.csv as kcsv
from kapture.io.features import get_descriptors_fullpath, image_descriptors_from_file
from kapture.io.features import get_keypoints_fullpath, image_keypoints_from_file
from kapture.io.records import TransferAction, get_image_fullpath

from kapture.converter.openmvg.import_openmvg import import_openmvg, import_openmvg_sfm_data_json  # noqa: E402
from kapture.converter.openmvg.export_openmvg import export_openmvg  # noqa: E402
from kapture.converter.openmvg.openmvg_commons import OPENMVG_SFM_DATA_VERSION_NUMBER, JSON_KEY
from kapture.converter.openmvg.openmvg_commons import CameraModel

logger = logging.getLogger('openmvg')


class TestOpenMvg(unittest.TestCase):

    # Constants related to the dataset to import
    FIRST_TRAJECTORY_TRANSLATION = [-2.1192500000000005, -0.4327849999999997, -4.07387]
    FIRST_TRAJECTORY_ROTATION = [0.9926160706165561, -0.08600940611886934, 0.0843934060039041, 0.01390940098954065]

    # Constants related to the dataset exported (M1X)
    FIRST_BASLER_SENSOR_ID = "22970285"
    FIRST_BASLER_OPENMVG_ID = 2147483649
    LIDAR1 = "lidar1"

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup variables for all the tests
        """
        logger.setLevel(logging.CRITICAL)
        cls.isWindows = sys.platform.startswith("win") or sys.platform.startswith("cygwin")
        test_file_path = path.dirname(__file__)
        cls._openmvg_sample_path = path.abspath(path.join(test_file_path, '../samples/t265/Everest_undistorted'))
        cls._kapture_sample_path = path.abspath(path.join(test_file_path, '../samples/m1x'))

    def setUp(self) -> None:
        """
        Setup before every test
        """
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_path = path.join(self._tempdir.name, 'from_openmvg')
        os.makedirs(self._kapture_path, exist_ok=True)

    def _verify_data(self, kapture_data) -> None:
        cameras = kapture_data.cameras
        self.assertIsNotNone(cameras, "Cameras exist")
        self.assertEqual(1, len(cameras), "One camera")
        camera = next(iter(cameras.values()))  # just take the first camera defined
        self.assertEqual(camera.camera_type, kapture.CameraType.SIMPLE_RADIAL_FISHEYE, "Type fisheye")
        camera_params = camera.camera_params
        self.assertEqual(848, camera_params[0], "width")
        self.assertEqual(800, camera_params[1], "height")
        records_camera = kapture_data.records_camera
        self.assertEqual(5, len(records_camera), "Number of images")
        first_record = records_camera[0]
        img_path = next(iter(first_record.values()))
        self.assertEqual("images/frame_000000001.jpg", img_path, "Image path")
        trajectories = kapture_data.trajectories
        self.assertEqual(5, len(trajectories), "Trajectories points")
        k_pose6d = next(iter(trajectories[0].values()))  # Kapture.PoseTransform
        ref_pose = kapture.PoseTransform(t=self.FIRST_TRAJECTORY_TRANSLATION, r=self.FIRST_TRAJECTORY_ROTATION)
        self.assertTrue(equal_poses(ref_pose, k_pose6d), "First trajectory pose")
        self.assertIsNone(kapture_data.keypoints, "No keypoints")
        self.assertIsNone(kapture_data.observations, "No observations")
        self.assertIsNone(kapture_data.points3d, "No 3D points")

    def test_openmvg_to_kapture(self) -> None:
        """
        Test the openmvg_to_kapture function on a small openmvg JSON file
        """
        json_file = path.join(self._openmvg_sample_path, 'sfm_data_small.json')
        with open(json_file, 'r') as f:
            sfm_data = json.load(f)
            views_id_to_filenames: Dict[int, str] = {}
            kapture_data = import_openmvg_sfm_data_json(sfm_data, self._kapture_path, views_id_to_filenames)
            self._verify_data(kapture_data)

    def test_import_openmvg(self) -> None:
        """
        Test the import_openmvg function on a small JSON file while linking the images
        """
        self.assertTrue(path.isdir(self._openmvg_sample_path))
        self.assertTrue(path.exists(self._kapture_path), "Kapture directory exists")
        sfm_file = path.join(self._openmvg_sample_path, 'sfm_data_small.json')
        # on windows, without admin rights, fails with OSError: symbolic link privilege not held
        # see https://docs.python.org/3.6/library/os.html#os.symlink
        logger.info(f'Running on "{sys.platform}" which is {"" if self.isWindows else "not"} a Windows platform')
        file_operation = TransferAction.skip if self.isWindows else TransferAction.link_relative
        import_openmvg(sfm_file, None, None, self._kapture_path, file_operation, True)
        #  test presence or absence of kapture files
        cameras_file_path = path.join(self._kapture_path, kcsv.CSV_FILENAMES[kapture.Sensors])
        self.assertTrue(path.isfile(cameras_file_path), "Camera file written")
        rigs_file_path = path.join(self._kapture_path, kcsv.CSV_FILENAMES[kapture.Rigs])
        self.assertFalse(path.isfile(rigs_file_path), "Rigs file should be missing")
        records_file_path = path.join(self._kapture_path, kcsv.CSV_FILENAMES[kapture.RecordsCamera])
        self.assertTrue(path.isfile(records_file_path), "Camera Records file written")
        lidars_file_path = path.join(self._kapture_path, kcsv.CSV_FILENAMES[kapture.RecordsLidar])
        self.assertFalse(path.isfile(lidars_file_path), "Lidar Records file should be missing")
        trajectories_file_path = path.join(self._kapture_path, kcsv.CSV_FILENAMES[kapture.Trajectories])
        self.assertTrue(path.isfile(trajectories_file_path), "Trajectories file written")
        # Reload data and verify
        kapture_data = kcsv.kapture_from_dir(self._kapture_path)
        self._verify_data(kapture_data)
        if not self.isWindows:
            # Test images path
            records_camera_files = kapture_data.records_camera.data_list()
            for name in records_camera_files:
                img_path = get_image_fullpath(self._kapture_path, name)
                self.assertTrue(path.islink(img_path), f"image link {img_path}")

    def test_kapture_to_openmvg(self) -> None:
        """
        Test the kapture_to_openmvg export function on a small kapture dataset
        """
        self.assertTrue(path.exists(self._kapture_sample_path), "Kapture directory exists")
        openmvg_json_file = path.join(self._tempdir.name, 'sfm_export.json')
        openmvg_image_root_path = path.join(self._tempdir.name, 'images')
        export_openmvg(kapture_path=self._kapture_sample_path,
                       openmvg_sfm_data_file_path=openmvg_json_file,
                       openmvg_image_root_path=openmvg_image_root_path,
                       image_action=TransferAction.copy,
                       force=True)
        self.assertTrue(path.isfile(openmvg_json_file), "Openmvg JSON file created")
        with open(openmvg_json_file, 'r') as f:
            sfm_data = json.load(f)
            self.assertEqual(OPENMVG_SFM_DATA_VERSION_NUMBER,
                             sfm_data.get(JSON_KEY.SFM_DATA_VERSION), "Sfm data version number")
            root_path = sfm_data.get(JSON_KEY.ROOT_PATH)
            self.assertIsNotNone(root_path, "Root path exported")
            self.assertEqual(openmvg_image_root_path, root_path, "Root path correct")
            intrinsics = sfm_data.get(JSON_KEY.INTRINSICS)
            self.assertIsNotNone(intrinsics, "Intrinsics")
            self.assertEqual(9, len(intrinsics), "Cameras")
            camera_ids = set()
            # Search for camera 22970285 and lidar1
            basler = None
            lidar = None
            for intrinsic in intrinsics:
                camera_id = intrinsic.get(JSON_KEY.VALUE, {}).get(JSON_KEY.PTR_WRAPPER, {}).get(JSON_KEY.ID)
                camera_ids.add(camera_id)
                if camera_id == self.FIRST_BASLER_OPENMVG_ID:
                    basler = intrinsic.get(JSON_KEY.VALUE)
                elif camera_id == self.LIDAR1:
                    lidar = intrinsic.get(JSON_KEY.VALUE)
            self.assertEqual(9, len(camera_ids), "All camera identifiers are different")
            self.assertIsNotNone(basler, "First basler camera")
            self.assertEqual(CameraModel.pinhole_brown_t2.name,
                             basler.get(JSON_KEY.POLYMORPHIC_NAME), "Polymorphic name")
            camera_params = basler.get(JSON_KEY.PTR_WRAPPER).get(JSON_KEY.DATA).get(JSON_KEY.VALUE0)
            self.assertEqual(2048, camera_params.get(JSON_KEY.WIDTH), "Camera width")
            self.assertEqual(1536, camera_params.get(JSON_KEY.HEIGHT), "Camera height")
            self.assertAlmostEqual(1725.842032333, camera_params.get(JSON_KEY.FOCAL_LENGTH), msg="Focal length")
            self.assertEqual(1024, camera_params.get(JSON_KEY.PRINCIPAL_POINT)[0], "Principal point X")
            self.assertEqual(768, camera_params.get(JSON_KEY.PRINCIPAL_POINT)[1], "Principal point Y")
            self.assertIsNotNone(basler.get(JSON_KEY.PTR_WRAPPER).get(JSON_KEY.DATA).get(JSON_KEY.DISTO_T2),
                                 "Disto_t2 defined")
            # Make sure lidar has not been exported to openMVG
            self.assertIsNone(lidar, "Lidar 1")
            # Recorded images: views in openmvg format
            views = sfm_data.get(JSON_KEY.VIEWS)
            self.assertIsNotNone(views, "Views")
            self.assertEqual(18, len(views), "Recorded images")
            image_record = None
            for view in views:
                if view.get(JSON_KEY.KEY) == 3:
                    image_record = view.get(JSON_KEY.VALUE).get(JSON_KEY.PTR_WRAPPER).get(JSON_KEY.DATA)
                    break
            self.assertIsNotNone(image_record, "4th image record")
            local_path = image_record.get(JSON_KEY.LOCAL_PATH)
            self.assertEqual(self.FIRST_BASLER_SENSOR_ID, local_path, "Local path is the camera id")
            self.assertEqual(camera_params.get(JSON_KEY.WIDTH), image_record.get(JSON_KEY.WIDTH),
                             "Image has camera width")
            self.assertEqual(camera_params.get(JSON_KEY.HEIGHT), image_record.get(JSON_KEY.HEIGHT),
                             "Image has camera height")
            filename = image_record.get(JSON_KEY.FILENAME)
            copied_image_path = path.join(root_path, local_path, filename)
            self.assertTrue(path.isfile(copied_image_path), "Image copied")
            self.assertIsNotNone(filename, "Filename is defined")
            pose_id = image_record.get(JSON_KEY.ID_POSE)
            self.assertIsNotNone(pose_id, "Pose id")
            self.assertTrue(image_record.get(JSON_KEY.USE_POSE_ROTATION_PRIOR), "Use pose rotation prior is true")
            view_center = image_record.get(JSON_KEY.CENTER)
            view_rotation = image_record.get(JSON_KEY.ROTATION)
            self.assertIsNotNone(view_center, "Center of image")
            self.assertIsNotNone(view_rotation, "Rotation of image")
            # poses: extrinsics in openmvg format
            extrinsics = sfm_data.get(JSON_KEY.EXTRINSICS)
            self.assertIsNotNone(extrinsics, "Extrinsics")
            self.assertEqual(10, len(extrinsics), "Trajectory points")
            # Search for the pose of the above image
            pose = None
            for extrinsic in extrinsics:
                if extrinsic.get(JSON_KEY.KEY) == pose_id:
                    pose = extrinsic.get(JSON_KEY.VALUE)
                    break
            self.assertIsNotNone(pose, "Pose for image")
            pose_center = pose.get(JSON_KEY.CENTER)
            pose_rotation = pose.get(JSON_KEY.ROTATION)
            self.assertEqual(view_center, pose_center, "Center are equal")
            self.assertEqual(view_rotation, pose_rotation, "Rotations are equal")

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self._tempdir.cleanup()


class TestOpenMvgReconstruction(unittest.TestCase):

    UNIQUE_CAMERA_OPENMVG_ID = 2147483649
    FIRST_IMAGE_NAME = "ChateauMaupertuisTest/MaupertuisTest_01.jpg"
    FIRST_TRAJECTORY_TRANSLATION = [-0.02839049268018459, 0.00255775313260552, 0.02473073308745868]
    FIRST_TRAJECTORY_ROTATION = [0.9999935675163499, 0.0004312835445997, 0.0033445836819604, -0.0012217530116397]
    FEATURE_TYPE = "SIFT"
    DESCRIPTOR_TYPE = FEATURE_TYPE+"_Image_describer"
    KEYPOINTS_TYPE = FEATURE_TYPE+"_Regions"

    @classmethod
    def setUpClass(cls) -> None:
        """
        Setup variables for all the tests
        """
        logger.setLevel(logging.CRITICAL)
        cls.isWindows = sys.platform.startswith("win") or sys.platform.startswith("cygwin")
        test_file_path = path.dirname(__file__)
        cls._openmvg_sample_path = path.abspath(path.join(test_file_path,
                                                          '../samples/maupertuis_openMVG/ChateauMaupertuisTest'))
        cls._kapture_sample_path = path.join(test_file_path,
                                             '../samples/maupertuis_openMVG/kapture_ChateauMaupertuisTest')
        cls._kapture_sample_path = path.abspath(cls._kapture_sample_path)

    def setUp(self) -> None:
        """
        Setup before every test
        """
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_path = path.join(self._tempdir.name, 'from_openmvg')
        os.makedirs(self._kapture_path, exist_ok=True)

    def _verify_data(self, kapture_data: kapture.Kapture, kapture_path: str) -> None:
        # Cameras
        cameras = kapture_data.cameras
        self.assertIsNotNone(cameras, "Cameras exist")
        self.assertEqual(1, len(cameras), "One camera")
        camera = next(iter(cameras.values()))  # just take the first camera defined
        self.assertEqual(camera.camera_type, kapture.CameraType.RADIAL, "Type Radial")
        camera_params = camera.camera_params
        self.assertEqual(1600, camera_params[0], "width")
        self.assertEqual(1200, camera_params[1], "height")
        # Images
        records_camera = kapture_data.records_camera
        self.assertEqual(4, len(records_camera), "Number of images")
        first_record = records_camera[0]
        img_path = next(iter(first_record.values()))
        self.assertEqual(self.FIRST_IMAGE_NAME, img_path, "Image path")
        # Poses
        trajectories = kapture_data.trajectories
        self.assertEqual(4, len(trajectories), "Trajectories points")
        k_pose6d = next(iter(trajectories[0].values()))  # Kapture.PoseTransform
        ref_pose = kapture.PoseTransform(t=self.FIRST_TRAJECTORY_TRANSLATION, r=self.FIRST_TRAJECTORY_ROTATION)
        self.assertTrue(equal_poses(ref_pose, k_pose6d), "First trajectory pose")
        # Descriptors
        kapture_descriptors = kapture_data.descriptors[self.DESCRIPTOR_TYPE]
        self.assertEqual(4, len(kapture_descriptors), "Descriptors")
        self.assertEqual(self.FEATURE_TYPE, kapture_descriptors.type_name, "Descriptors feature type")
        self.assertEqual(self.KEYPOINTS_TYPE, kapture_descriptors.keypoints_type, "Descriptors keypoints type")
        self.assertIs(numpy.uint8, kapture_descriptors.dtype, "Descriptors dtype")
        self.assertEqual(128, kapture_descriptors.dsize, "Descriptors dsize")
        descriptors_file_path = get_descriptors_fullpath(self.DESCRIPTOR_TYPE,
                                                         self._kapture_path,
                                                         self.FIRST_IMAGE_NAME)
        first_descriptors_data = image_descriptors_from_file(descriptors_file_path,
                                                             kapture_descriptors.dtype,
                                                             kapture_descriptors.dsize)
        self.assertEqual((1850, 128), first_descriptors_data.shape, "Descriptors shape")
        self.assertEqual(numpy.uint8, first_descriptors_data.dtype, "Descriptors dtype")

        # Keypoints
        kapture_keypoints = kapture_data.keypoints[self.KEYPOINTS_TYPE]
        self.assertEqual(4, len(kapture_keypoints), "Keypoints")
        self.assertEqual(self.FEATURE_TYPE, kapture_keypoints.type_name, "Keypoints feature type")
        self.assertIs(float, kapture_keypoints.dtype, "Keypoints dtype")
        self.assertEqual(4, kapture_keypoints.dsize, "Keypoints dsize")
        keypoints_file_path = get_keypoints_fullpath(self.KEYPOINTS_TYPE,
                                                     kapture_path,
                                                     self.FIRST_IMAGE_NAME,
                                                     None)
        first_keypoints_data = image_keypoints_from_file(keypoints_file_path,
                                                         kapture_keypoints.dtype,
                                                         kapture_keypoints.dsize)
        self.assertEqual(first_keypoints_data.shape, (1850, 4), "Keypoints shape")
        self.assertEqual(first_keypoints_data.dtype, numpy.float64, "Keypoints dtype")

        # Observations
        kapture_observations = kapture_data.observations
        self.assertEqual(len(kapture_observations), 365, "Observations")
        self.assertEqual(kapture_observations.observations_number(), 1228, "Number of observations")
        image_name_subdir = path.dirname(self.FIRST_IMAGE_NAME)
        for per_feature_observations in kapture_observations.values():
            for keypoints_type, observations_list in per_feature_observations.items():
                self.assertEqual(keypoints_type, self.KEYPOINTS_TYPE, "Observation keypoints type")
                self.assertLessEqual(len(observations_list), 4, "observations per 3D point")
                for observation in observations_list:
                    self.assertTrue(observation[0].startswith(image_name_subdir), "Observation image name subdirectory")
        # 3D Points
        kapture_points3d = kapture_data.points3d
        self.assertEqual(kapture_points3d.shape, (372, kapture.Points3d.XYZ_ONLY), "3D points shape")

    def test_import_openmvg_reconstruction(self) -> None:
        """
        Test the import_openmvg function on a reconstruction JSON file with keypoints, 3D points and observations
        """
        self.assertTrue(path.isdir(self._openmvg_sample_path))
        self.assertTrue(path.exists(self._kapture_path), "Kapture directory exists")
        sfm_file = path.join(self._openmvg_sample_path, 'reconstruction_global', 'sfm_data.json')
        openmvg_matches_dir = path.join(self._openmvg_sample_path, 'matches')
        openmvg_matches_file = path.join(openmvg_matches_dir, 'matches.f.txt')
        import_openmvg(sfm_file, openmvg_matches_dir, openmvg_matches_file, self._kapture_path, TransferAction.skip)
        #  test presence or absence of kapture files
        cameras_file_path = kcsv.get_csv_fullpath(kapture.Sensors, self._kapture_path)
        self.assertTrue(path.isfile(cameras_file_path), "Camera file written")
        rigs_file_path = kcsv.get_csv_fullpath(kapture.Rigs, self._kapture_path)
        self.assertFalse(path.isfile(rigs_file_path), "Rigs file should be missing")
        records_file_path = kcsv.get_csv_fullpath(kapture.RecordsCamera, self._kapture_path)
        self.assertTrue(path.isfile(records_file_path), "Camera Records file written")
        lidars_file_path = kcsv.get_csv_fullpath(kapture.RecordsLidar, self._kapture_path)
        self.assertFalse(path.isfile(lidars_file_path), "Lidar Records file should be missing")
        trajectories_file_path = kcsv.get_csv_fullpath(kapture.Trajectories, self._kapture_path)
        self.assertTrue(path.isfile(trajectories_file_path), "Trajectories file written")
        keypoints_file_path = kcsv.get_feature_csv_fullpath(kapture.Keypoints, self.KEYPOINTS_TYPE, self._kapture_path)
        self.assertTrue(path.isfile(keypoints_file_path), "keypoints file written")
        descriptors_file_path = kcsv.get_feature_csv_fullpath(kapture.Descriptors,
                                                              "SIFT_Image_describer",
                                                              self._kapture_path)
        self.assertTrue(path.isfile(descriptors_file_path), "descriptors file written")
        points3d_file_path = kcsv.get_csv_fullpath(kapture.Points3d, self._kapture_path)
        self.assertTrue(path.isfile(points3d_file_path), "3D points file written")
        observations_file_path = kcsv.get_csv_fullpath(kapture.Observations, self._kapture_path)
        self.assertTrue(path.isfile(observations_file_path), "observations file written")

        # Reload data and verify
        kapture_data = kcsv.kapture_from_dir(self._kapture_path)
        kapture_sample_data = kcsv.kapture_from_dir(self._kapture_sample_path)
        self.assertTrue(equal_kapture(kapture_sample_data, kapture_data), "Created kapture is equal to sample")
        # Compare with kapture sample
        self._verify_data(kapture_data, self._kapture_path)

    def test_kapture_to_openmvg_reconstruction(self) -> None:
        """
        Test the kapture_to_openmvg export function on a small kapture dataset with 3D reconstruction data
        """
        self.assertTrue(path.exists(self._kapture_sample_path), "Kapture directory exists")
        kapture_sample_data = kcsv.kapture_from_dir(self._kapture_sample_path)
        openmvg_json_file = path.join(self._tempdir.name, 'sfm_export_3d.json')
        openmvg_image_root_path = path.join(self._tempdir.name, 'images')
        regions_dir_path = path.join(self._tempdir.name, 'regions')
        export_openmvg(kapture_path=self._kapture_sample_path,
                       openmvg_sfm_data_file_path=openmvg_json_file,
                       openmvg_image_root_path=openmvg_image_root_path,
                       openmvg_regions_dir_path=regions_dir_path,
                       image_action=TransferAction.copy,
                       force=True)
        self.assertTrue(path.isfile(openmvg_json_file), "Openmvg JSON file created")
        with open(openmvg_json_file, 'r') as f:
            sfm_data = json.load(f)
            self.assertEqual(OPENMVG_SFM_DATA_VERSION_NUMBER,
                             sfm_data.get(JSON_KEY.SFM_DATA_VERSION), "Sfm data version number")
            root_path: str = sfm_data.get(JSON_KEY.ROOT_PATH)
            self.assertIsNotNone(root_path, "Root path exported")
            self.assertTrue(root_path.startswith(openmvg_image_root_path), "Root path correct")
            kapture_camera: kapture.Camera = kapture_sample_data.cameras.get('0')
            kapture_camera_params = kapture_camera.camera_params
            intrinsics = sfm_data.get(JSON_KEY.INTRINSICS)
            self.assertIsNotNone(intrinsics, "Intrinsics")
            self.assertEqual(1, len(intrinsics), "Camera")
            intrinsic = intrinsics[0]
            unique_camera = intrinsic.get(JSON_KEY.VALUE)
            self.assertIsNotNone(unique_camera, "Unique camera")
            self.assertEqual(CameraModel.pinhole_radial_k3.name,
                             unique_camera.get(JSON_KEY.POLYMORPHIC_NAME), "Polymorphic name")
            camera_id = unique_camera.get(JSON_KEY.PTR_WRAPPER, {}).get(JSON_KEY.ID)
            self.assertEqual(self.UNIQUE_CAMERA_OPENMVG_ID, camera_id, "Unique openmvg camera id")
            camera_params = unique_camera.get(JSON_KEY.PTR_WRAPPER, {}).get(JSON_KEY.DATA)
            self.assertIsNotNone(camera_params, "Camera params")
            self.assertEqual(kapture_camera_params[0], camera_params.get(JSON_KEY.WIDTH), "Camera width")
            self.assertEqual(kapture_camera_params[1], camera_params.get(JSON_KEY.HEIGHT), "Camera height")
            self.assertAlmostEqual(kapture_camera_params[2], camera_params.get(JSON_KEY.FOCAL_LENGTH), msg="Focal len")
            principal_point = camera_params.get(JSON_KEY.PRINCIPAL_POINT)
            self.assertEqual(kapture_camera_params[3], principal_point[0], "Principal point X")
            self.assertEqual(kapture_camera_params[4], principal_point[1], "Principal point Y")
            disto_k3 = camera_params.get(JSON_KEY.DISTO_K3)
            self.assertIsNotNone(disto_k3, "Disto_k3 defined")
            self.assertEqual(kapture_camera_params[5], disto_k3[0], "Disto K1")
            self.assertEqual(kapture_camera_params[6], disto_k3[1], "Disto K2")
            # Recorded images: views in openmvg format
            views = sfm_data.get(JSON_KEY.VIEWS)
            self.assertIsNotNone(views, "Views")
            self.assertEqual(4, len(views), "Recorded images")
            image_record = None
            for view in views:
                if view.get(JSON_KEY.KEY) == 0:
                    image_record = view.get(JSON_KEY.VALUE).get(JSON_KEY.PTR_WRAPPER).get(JSON_KEY.DATA)
                    break
            self.assertIsNotNone(image_record, "1th image record")
            local_path = image_record.get(JSON_KEY.LOCAL_PATH)
            self.assertEqual("", local_path, "Local path is empty")
            self.assertEqual(camera_params.get(JSON_KEY.WIDTH), image_record.get(JSON_KEY.WIDTH),
                             "Image has camera width")
            self.assertEqual(camera_params.get(JSON_KEY.HEIGHT), image_record.get(JSON_KEY.HEIGHT),
                             "Image has camera height")
            filename = image_record.get(JSON_KEY.FILENAME)
            self.assertIsNotNone(filename, "Filename is defined")
            copied_image_path = path.join(root_path, local_path, filename)
            self.assertTrue(path.isfile(copied_image_path), "Image copied")
            pose_id = image_record.get(JSON_KEY.ID_POSE)
            self.assertIsNotNone(pose_id, "Pose id")
            self.assertTrue(image_record.get(JSON_KEY.USE_POSE_ROTATION_PRIOR), "Use pose rotation prior is true")
            view_center = image_record.get(JSON_KEY.CENTER)
            view_rotation = image_record.get(JSON_KEY.ROTATION)
            self.assertIsNotNone(view_center, "Center of image")
            self.assertIsNotNone(view_rotation, "Rotation of image")
            # poses: extrinsics in openmvg format
            extrinsics = sfm_data.get(JSON_KEY.EXTRINSICS)
            self.assertIsNotNone(extrinsics, "Extrinsics")
            self.assertEqual(4, len(extrinsics), "Trajectory points")
            # Search for the pose of the above image
            pose = None
            for extrinsic in extrinsics:
                if extrinsic.get(JSON_KEY.KEY) == pose_id:
                    pose = extrinsic.get(JSON_KEY.VALUE)
                    break
            self.assertIsNotNone(pose, "Pose for image")
            pose_center = pose.get(JSON_KEY.CENTER)
            pose_rotation = pose.get(JSON_KEY.ROTATION)
            self.assertEqual(view_center, pose_center, "Center are equal")
            self.assertEqual(view_rotation, pose_rotation, "Rotations are equal")
            # Check the openmvg structure
            openmvg_structure = sfm_data.get(JSON_KEY.STRUCTURE)
            kapture_points3d = kapture_sample_data.points3d
            nb_3d_points = kapture_points3d.shape[0]
            self.assertEqual(nb_3d_points, len(openmvg_structure), "All 3D points in openmvg structure")
            # check that all keys are different
            point_idx_set = set()
            for point_3d_structure in openmvg_structure:
                point_idx = point_3d_structure.get(JSON_KEY.KEY)
                self.assertIsNotNone(point_idx, "Point 3D index")
                point_idx_set.add(point_idx)
                point3d_value = point_3d_structure.get(JSON_KEY.VALUE)
                self.assertIsNotNone(point3d_value)
                coords: List[float] = point3d_value.get(JSON_KEY.X)
                self.assertIsNotNone(coords, "3D coordinates")
                observations = point3d_value.get(JSON_KEY.OBSERVATIONS)
                self.assertIsNotNone(observations, "Observations")
            self.assertEqual(nb_3d_points, len(point_idx_set), "All point 3D indexes are different")
            # Check that features and descriptors files are there
            kaptures_images_files = kapture_sample_data.records_camera.data_list()
            for kapture_image_file in kaptures_images_files:
                image_file_name = path.splitext(path.basename(kapture_image_file))[0]
                feature_image_file = path.join(regions_dir_path, image_file_name)+'.feat'
                self.assertTrue(path.isfile(feature_image_file), f"Features file for {image_file_name}")
                descriptor_image_file = path.join(regions_dir_path, image_file_name)+'.desc'
                self.assertTrue(path.isfile(descriptor_image_file), f"Descriptors file for {image_file_name}")

    def test_kapture_export_to_openmvg_reconstruction(self) -> None:
        """
          Test the export_openmvg function on a small kapture dataset with 3D reconstruction data
          by doing a round trip kapture -> openmvg -> kapture conversion
        """

        self.assertTrue(path.exists(self._kapture_sample_path), "Kapture directory exists")
        openmvg_json_file = path.join(self._tempdir.name, 'sfm_export_3d.json')
        openmvg_image_root_path = path.join(self._tempdir.name, 'images')
        regions_dir_path = path.join(self._tempdir.name, 'regions')
        matches_file_path = path.join(self._tempdir.name, 'matches', 'matches.f.txt')
        export_openmvg(self._kapture_sample_path,
                       openmvg_json_file,
                       openmvg_image_root_path,
                       regions_dir_path,
                       matches_file_path,
                       TransferAction.copy)
        self.assertTrue(path.isfile(openmvg_json_file), "Openmvg JSON file created")
        self.assertTrue(path.isdir(regions_dir_path), "Openmvg regions dir created")
        self.assertTrue(path.isfile(matches_file_path), "Openmvg matches file")
        # Reimport the exported data
        import_openmvg(openmvg_json_file, regions_dir_path, matches_file_path, self._kapture_path, TransferAction.copy)
        kapture_data = kcsv.kapture_from_dir(self._kapture_path)
        # Compare to the original sample
        kapture_sample_data = kcsv.kapture_from_dir(self._kapture_sample_path)
        self.assertTrue(equal_kapture(kapture_sample_data, kapture_data), "Kaptures are equal")

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self._tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
