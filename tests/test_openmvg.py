#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import json
import logging
import sys
import os
import os.path as path
import tempfile
import unittest
# kapture
import path_to_kapture  # enables import kapture
import kapture
from kapture.algo.compare import equal_poses
import kapture.io.csv as kcsv
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.converter.openmvg.import_openmvg import openmvg_to_kapture, import_openmvg  # noqa: E402
from kapture.converter.openmvg.export_openmvg import export_openmvg  # noqa: E402
from kapture.converter.openmvg.openmvg_commons import SFM_DATA_VERSION,\
    SFM_DATA_VERSION_NUMBER, ROOT_PATH, INTRINSICS, VIEWS, EXTRINSICS, KEY, VALUE,\
    PTR_WRAPPER, DATA, LOCAL_PATH, FILENAME, ID_INTRINSIC, ID_POSE, POLYMORPHIC_NAME, VALUE0,\
    WIDTH, HEIGHT, FOCAL_LENGTH, PRINCIPAL_POINT, DISTO_T2,\
    CENTER, USE_POSE_ROTATION_PRIOR, ROTATION
from kapture.converter.openmvg.openmvg_commons import CameraModel


logger = logging.getLogger('openmvg')

# Constants related to the dataset to import
FIRST_TRAJECTORY_TRANSLATION = [-2.1192500000000005, -0.4327849999999997, -4.07387]
FIRST_TRAJECTORY_ROTATION = [0.9926160706165561, -0.08600940611886934, 0.0843934060039041, 0.01390940098954065]

# Constants related to the dataset exported (M1X)
FIRST_BASLER_ID = "22970285"
LIDAR1 = "lidar1"


class TestOpenMvg(unittest.TestCase):

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
        ref_pose = kapture.PoseTransform(t=FIRST_TRAJECTORY_TRANSLATION, r=FIRST_TRAJECTORY_ROTATION)
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
            kapture_data = openmvg_to_kapture(sfm_data, self._kapture_path)
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
        import_openmvg(sfm_file, self._kapture_path, file_operation, True)
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
            all_records_camera = list(kapture.flatten(kapture_data.records_camera))
            for _, _, name in all_records_camera:
                img_path = get_image_fullpath(self._kapture_path, name)
                self.assertTrue(path.islink(img_path), f"image link {img_path}")

    def test_kapture_to_openmvg(self) -> None:
        """
        Test the kapture_to_openmvg export function on a small kapture dataset
        """
        self.assertTrue(path.exists(self._kapture_sample_path), "Kapture directory exists")
        json_file = path.join(self._tempdir.name, 'sfm_export.json')
        export_openmvg(self._kapture_sample_path, json_file, TransferAction.copy, force=True)
        self.assertTrue(path.isfile(json_file), "Openmvg JSON file created")
        with open(json_file, 'r') as f:
            sfm_data = json.load(f)
            self.assertEqual(SFM_DATA_VERSION_NUMBER, sfm_data.get(SFM_DATA_VERSION), "Sfm data version number")
            root_path = sfm_data.get(ROOT_PATH)
            self.assertIsNotNone(root_path, "Root path exported")
            self.assertEqual(self._tempdir.name, root_path, "Root path correct")
            intrinsics = sfm_data.get(INTRINSICS)
            self.assertIsNotNone(intrinsics, "Intrinsics")
            self.assertEqual(9, len(intrinsics), "Cameras")
            camera_ids = {}
            # Search for camera 22970285 and lidar1
            basler = None
            lidar = None
            for intrinsic in intrinsics:
                camera_id = intrinsic.get(KEY)
                camera_ids[camera_id] = camera_id
                if camera_id == FIRST_BASLER_ID:
                    basler = intrinsic.get(VALUE)
                elif camera_id == LIDAR1:
                    lidar = intrinsic.get(VALUE)
            self.assertEqual(9, len(camera_ids), "All camera identifiers are different")
            self.assertIsNotNone(basler, "First basler camera")
            self.assertEqual(CameraModel.pinhole_brown_t2.name, basler.get(POLYMORPHIC_NAME), "Polymorphic name")
            camera_params = basler.get(PTR_WRAPPER).get(DATA).get(VALUE0)
            self.assertEqual(2048, camera_params.get(WIDTH), "Camera width")
            self.assertEqual(1536, camera_params.get(HEIGHT), "Camera height")
            self.assertAlmostEqual(1725.842032333, camera_params.get(FOCAL_LENGTH), msg="Focal length")
            self.assertEqual(1024, camera_params.get(PRINCIPAL_POINT)[0], "Principal point X")
            self.assertEqual(768, camera_params.get(PRINCIPAL_POINT)[1], "Principal point Y")
            self.assertIsNotNone(basler.get(PTR_WRAPPER).get(DATA).get(DISTO_T2), "Disto_t2 defined")
            # Make sure lidar has not been exported to openMVG
            self.assertIsNone(lidar, "Lidar 1")
            # Recorded images: views in openmvg format
            views = sfm_data.get(VIEWS)
            self.assertIsNotNone(views, "Views")
            self.assertEqual(18, len(views), "Recorded images")
            image_record = None
            for view in views:
                if view.get(KEY) == 3:
                    image_record = view.get(VALUE).get(PTR_WRAPPER).get(DATA)
                    break
            self.assertIsNotNone(image_record, "4th image record")
            local_path = image_record.get(LOCAL_PATH)
            self.assertEqual(FIRST_BASLER_ID, local_path, "Local path is the camera id")
            self.assertEqual(FIRST_BASLER_ID, image_record.get(ID_INTRINSIC), "id_intrinsic is the camera id")
            self.assertEqual(camera_params.get(WIDTH), image_record.get(WIDTH), "Image has camera width")
            self.assertEqual(camera_params.get(HEIGHT), image_record.get(HEIGHT), "Image has camera height")
            filename = image_record.get(FILENAME)
            copied_image_path = path.join(root_path, local_path, filename)
            self.assertTrue(path.isfile(copied_image_path), "Image copied")
            self.assertIsNotNone(filename, "Filename is defined")
            pose_id = image_record.get(ID_POSE)
            self.assertIsNotNone(pose_id, "Pose id")
            self.assertTrue(image_record.get(USE_POSE_ROTATION_PRIOR), "Use pose rotation prior is true")
            view_center = image_record.get(CENTER)
            view_rotation = image_record.get(ROTATION)
            self.assertIsNotNone(view_center, "Center of image")
            self.assertIsNotNone(view_rotation, "Rotation of image")
            # poses: extrinsics in openmvg format
            extrinsics = sfm_data.get(EXTRINSICS)
            self.assertIsNotNone(extrinsics, "Extrinsics")
            self.assertEqual(10, len(extrinsics), "Trajectory points")
            # Search for the pose of the above image
            pose = None
            for extrinsic in extrinsics:
                if extrinsic.get(KEY) == pose_id:
                    pose = extrinsic.get(VALUE)
                    break
            self.assertIsNotNone(pose, "Pose for image")
            pose_center = pose.get(CENTER)
            pose_rotation = pose.get(ROTATION)
            self.assertEqual(view_center, pose_center, "Center are equal")
            self.assertEqual(view_rotation, pose_rotation, "Rotations are equal")

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self._tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
