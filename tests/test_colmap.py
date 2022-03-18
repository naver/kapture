#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os.path as path
import tempfile
import numpy as np
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.io.csv import kapture_from_dir
from kapture.io.features import image_keypoints_from_file, image_descriptors_from_file
from kapture.algo.compare import equal_poses
from kapture.utils.paths import path_secure
# tools
from kapture.converter.colmap.import_colmap import import_colmap, import_colmap_database, \
    import_colmap_from_reconstruction_files  # noqa: E402
from kapture.converter.colmap.export_colmap import export_colmap
from kapture.converter.colmap.export_colmap_rigs import export_colmap_rig_json
from kapture.converter.colmap.import_colmap_rigs import import_colmap_rig_json
from kapture.converter.colmap.cameras import get_camera_kapture_id_from_colmap_id
from kapture.converter.colmap.database import COLMAPDatabase
from kapture.converter.colmap.database_extra import exists_table

here_dirpath = path.abspath(path.dirname(__file__))


class TestColmapPrimitives(unittest.TestCase):
    def test_exists_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            colmap_db_path = path.join(tmpdirname, 'colmap.db')
            colmap_db = COLMAPDatabase.connect(colmap_db_path)
            self.assertFalse(exists_table('two_view_geometries', colmap_db))
            colmap_db.create_tables()
            self.assertTrue(exists_table('two_view_geometries', colmap_db))
            colmap_db.close()


class TestColmapImportT265(unittest.TestCase):
    def setUp(self) -> None:
        self._sample_dirpath = path.join(here_dirpath, '..', 'samples', 't265', 'colmap')
        self._tempdir = tempfile.TemporaryDirectory()
        self._database_filepath = path.join(self._sample_dirpath, 'colmap.db')
        self._reconstruction_path = self._sample_dirpath
        self._kapture_dirpath = path.join(self._tempdir.name, 'kapture')

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_t265_db_only(self):
        kapture_data = import_colmap_database(self._database_filepath, self._kapture_dirpath,
                                              no_geometric_filtering=True)

        # check the numbers
        self.assertEqual(2, len(kapture_data.sensors))
        self.assertEqual(6, len(kapture_data.trajectories))
        self.assertEqual(6, len(kapture_data.records_camera))

        # check camera ids
        camera_ids_expected = set(['cam_00001', 'cam_00002'])  # may evolve in future, not crucial
        camera_ids_actual = set(kapture_data.sensors.keys())
        self.assertEqual(camera_ids_expected, camera_ids_actual)
        # check camera ids consistent in trajectories
        camera_ids_trajectories = set(cam_id for _, cam_id, _ in kapture.flatten(kapture_data.trajectories))
        self.assertEqual(camera_ids_actual, camera_ids_trajectories)
        # check camera ids consistent in records_camera
        camera_ids_records = set(cam_id for _, cam_id, _ in kapture.flatten(kapture_data.records_camera))
        self.assertEqual(camera_ids_actual, camera_ids_records)

        # check camera parameters
        cam1 = kapture_data.sensors['cam_00001']
        self.assertIsInstance(cam1, kapture.Camera)
        self.assertEqual(kapture.SensorType.camera.name, cam1.sensor_type)
        self.assertEqual(kapture.CameraType.OPENCV_FISHEYE, cam1.camera_type)
        params_expected = [848.0, 800.0, 284.468, 285.51, 424.355, 393.742, 0.0008, 0.031, -0.03, 0.005]
        self.assertAlmostEqual(params_expected, cam1.camera_params)

        # check records
        timestamp, cam_id, image = next(kapture.flatten(kapture_data.records_camera, is_sorted=True))
        self.assertEqual(1, timestamp)
        self.assertEqual('cam_00002', cam_id)
        self.assertEqual('rightraw/frame_000000001.jpg', image)

        # check trajectories
        timestamp, cam_id, pose = next(kapture.flatten(kapture_data.trajectories, is_sorted=True))
        self.assertEqual(1, timestamp)
        self.assertEqual('cam_00002', cam_id)
        pose_expected = kapture.PoseTransform(r=[0.9540331248716523, -0.03768128483784883, -0.2972570621910482,
                                                 -0.0062565444214723875],
                                              t=[2.7109402281860904, 0.13236653865769618, -2.868626176500939])
        self.assertTrue(equal_poses(pose_expected, pose))

        # this sample has no keypoints, descriptors nor matches
        self.assertFalse(path.exists(path.join(self._kapture_dirpath, 'reconstruction')))


class TestColmapMaupertuis(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._temp_dirpath = self._tempdir.name
        self._sample_dirpath = path.join(here_dirpath, '..', 'samples', 'maupertuis')
        self._colmap_dirpath = path.join(self._sample_dirpath, 'colmap')
        self._images_filepath = path.join(self._colmap_dirpath, 'images')
        self._database_filepath = path.join(self._colmap_dirpath, 'colmap.db')
        self._reconstruction_path = path.join(self._colmap_dirpath, 'reconstruction')
        self._kapture_dirpath = path.normpath(path.join(self._sample_dirpath, 'kapture'))
        self.keypoints_type = 'SIFT'
        self.descriptors_type = 'SIFT'

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def test_maupertuis_import_db_only(self):
        kapture_data = import_colmap_database(self._database_filepath, self._temp_dirpath,
                                              self.keypoints_type,
                                              self.descriptors_type,
                                              no_geometric_filtering=True)

        # check the numbers
        self.assertIsNone(kapture_data.trajectories)
        self.assertIsNone(kapture_data.points3d)
        self.assertIsNone(kapture_data.records_lidar)
        self.assertIsNone(kapture_data.records_wifi)
        self.assertIsNone(kapture_data.records_gnss)
        self.assertEqual(1, len(kapture_data.sensors))
        self.assertEqual(4, len(kapture_data.records_camera))
        self.assertEqual(4, len(kapture_data.keypoints[self.keypoints_type]))
        self.assertEqual(4, len(kapture_data.descriptors[self.descriptors_type]))
        self.assertEqual(6, len(kapture_data.matches[self.keypoints_type]))

        # check camera
        camera = kapture_data.sensors['cam_00001']
        self.assertEqual(kapture.SensorType.camera.name, camera.sensor_type)
        self.assertEqual(kapture.CameraType.SIMPLE_PINHOLE, camera.camera_type)
        self.assertAlmostEqual(camera.camera_params, [1919.0, 1079.0, 2302.7999999999997, 959.5, 539.5])

        # check snapshots
        snapshots = kapture_data.records_camera
        self.assertTrue(all('cam_00001' in ts for ts in snapshots.values()))
        self.assertEqual(['00.jpg', '01.jpg', '02.jpg', '03.jpg'],
                         [filename for _, _, filename in kapture.flatten(snapshots, True)])

        # check keypoints
        keypoints = kapture_data.keypoints[self.keypoints_type]
        self.assertEqual(np.float32, keypoints.dtype)
        self.assertEqual(6, keypoints.dsize)
        self.assertEqual({'00.jpg', '01.jpg', '02.jpg', '03.jpg'}, keypoints)
        keypoints_filepaths_actual = kapture.io.features.keypoints_to_filepaths(keypoints,
                                                                                self.keypoints_type,
                                                                                self._temp_dirpath)
        keypoints_filepaths_expected = {
            f'{i:02d}.jpg': path_secure(f'{self._temp_dirpath}/reconstruction/keypoints'
                                        f'/{self.keypoints_type}/{i:02d}.jpg.kpt')
            for i in [0, 1, 2, 3]}
        self.assertDictEqual(keypoints_filepaths_actual, keypoints_filepaths_expected)
        # check a keypoints file
        image_keypoints_filepaths = sorted(
            kapture.io.features.keypoints_to_filepaths(keypoints, self.keypoints_type, self._temp_dirpath).values())
        image_keypoints = image_keypoints_from_file(image_keypoints_filepaths[0], keypoints.dtype, keypoints.dsize)
        self.assertEqual((6424, 6), image_keypoints.shape)
        self.assertAlmostEqual([1290.908447265625, 4.156360626220703, -1.3475048542022705,
                                1.4732409715652466, -1.4732409715652466, -1.3475048542022705],
                               image_keypoints[0].tolist())

        self.assertAlmostEqual([1381.316650390625, 668.8056640625, 59.981021881103516,
                                46.423213958740234, -46.423213958740234, 59.981021881103516],
                               image_keypoints[-1].tolist())

        # check descriptors
        descriptors = kapture_data.descriptors[self.descriptors_type]
        self.assertEqual(np.uint8, descriptors.dtype)
        self.assertEqual(128, descriptors.dsize)
        self.assertEqual({'00.jpg', '01.jpg', '02.jpg', '03.jpg'}, descriptors)
        descriptors_filepaths_actual = kapture.io.features.descriptors_to_filepaths(descriptors,
                                                                                    self.descriptors_type,
                                                                                    self._temp_dirpath)
        descriptors_filepaths_expected = {
            f'{i:02d}.jpg': path_secure(f'{self._temp_dirpath}/reconstruction/descriptors/'
                                        f'{self.descriptors_type}/{i:02d}.jpg.desc')
            for i in [0, 1, 2, 3]}
        self.assertDictEqual(descriptors_filepaths_actual, descriptors_filepaths_expected)
        # check a descriptors file
        image_descriptors_filepaths = sorted(kapture.io.features.descriptors_to_filepaths(descriptors,
                                                                                          self.descriptors_type,
                                                                                          self._temp_dirpath).values())
        image_descriptors = image_descriptors_from_file(
            image_descriptors_filepaths[0], descriptors.dtype, descriptors.dsize)
        self.assertEqual(image_keypoints.shape[0], image_descriptors.shape[0])

        # check matches
        matches = kapture_data.matches[self.keypoints_type]
        self.assertEqual({('01.jpg', '03.jpg'), ('00.jpg', '02.jpg'),
                          ('00.jpg', '03.jpg'), ('02.jpg', '03.jpg'),
                          ('00.jpg', '01.jpg'), ('01.jpg', '02.jpg')},
                         set(matches))

    def test_maupertuis_import_txt_only(self):
        kapture_data = import_colmap_from_reconstruction_files(
            self._reconstruction_path, self._temp_dirpath, self.keypoints_type, skip=set())

        # check the numbers
        self.assertEqual(1, len(kapture_data.sensors))
        self.assertEqual(4, len(kapture_data.trajectories))
        self.assertEqual(4, len(kapture_data.records_camera))
        self.assertIs(kapture_data.records_lidar, None)
        self.assertIs(kapture_data.records_wifi, None)
        self.assertEqual(4, len(kapture_data.keypoints[self.keypoints_type]))
        self.assertIs(kapture_data.descriptors, None)
        self.assertIs(kapture_data.matches, None)
        self.assertEqual(1039, len(kapture_data.points3d))
        self.assertEqual(1039, len(kapture_data.observations))

        # check camera
        camera = kapture_data.sensors['cam_00001']
        self.assertEqual(kapture.SensorType.camera.name, camera.sensor_type)
        self.assertEqual(kapture.CameraType.SIMPLE_PINHOLE, camera.camera_type)
        self.assertAlmostEqual(camera.camera_params, [1919.0, 1079.0, 1847.53, 959.5, 539.5])

        # check snapshots
        snapshots = kapture_data.records_camera
        self.assertTrue(all('cam_00001' in ts for ts in snapshots.values()))
        self.assertEqual(['00.jpg', '01.jpg', '02.jpg', '03.jpg'],
                         [filename for _, _, filename in kapture.flatten(snapshots, True)])

        # check trajectories
        trajectory = kapture_data.trajectories
        self.assertTrue(all('cam_00001' in ts for ts in trajectory.values()))
        self.assertTrue(all(pose.r is not None and pose.t is not None
                            for ts in trajectory.values()
                            for pose in ts.values()))
        self.assertTrue(equal_poses(trajectory[1, 'cam_00001'],
                                    kapture.PoseTransform(r=[0.998245, -0.000889039, -0.0384732, -0.045019],
                                                          t=[3.24777, -2.58119, -0.0457181])))

        # check points3d
        self.assertEqual((1039, 6), kapture_data.points3d.shape)
        self.assertAlmostEqual([-2.39675, 4.62278, 13.2759, 57.0, 57.0, 49.0],
                               kapture_data.points3d[0].tolist())

        # check observations
        observations = kapture_data.observations
        # self.assertEqual(4, len(observations[0]))
        self.assertEqual({('01.jpg', 4561), ('02.jpg', 3389), ('00.jpg', 4975), ('03.jpg', 3472)},
                         set(observations[0]['SIFT']))

    def test_maupertuis_import(self):
        kapture_data = import_colmap(self._temp_dirpath, self._database_filepath,
                                     self._reconstruction_path, self._images_filepath,
                                     None, self.keypoints_type, self.descriptors_type,
                                     force_overwrite_existing=True,
                                     images_import_strategy=TransferAction.copy,
                                     no_geometric_filtering=True)

        # check the numbers
        self.assertEqual(1, len(kapture_data.sensors))
        self.assertEqual(4, len(kapture_data.trajectories))
        self.assertEqual(4, len(kapture_data.records_camera))
        self.assertIs(kapture_data.records_lidar, None)
        self.assertIs(kapture_data.records_wifi, None)
        self.assertIs(kapture_data.records_gnss, None)
        self.assertEqual(4, len(kapture_data.keypoints[self.keypoints_type]))
        self.assertEqual(4, len(kapture_data.descriptors[self.descriptors_type]))
        self.assertEqual(6, len(kapture_data.matches[self.keypoints_type]))
        self.assertEqual(1039, len(kapture_data.points3d))
        self.assertEqual(1039, len(kapture_data.observations))

        # compare against golden kapture
        kapture_data_golden = kapture_from_dir(self._kapture_dirpath)
        # drop GPS, Wifi, Lidar
        kapture_data.records_lidar = None
        kapture_data.records_wifi = None
        kapture_data_golden.records_gnss = None
        kapture_data_golden.sensors = kapture.Sensors({
            sensor_id: sensor
            for sensor_id, sensor in kapture_data_golden.sensors.items()
            if isinstance(sensor, kapture.Camera)
        })

        # compare
        equivalence = kapture.algo.compare.equal_kapture(kapture_data, kapture_data_golden)
        self.assertTrue(equivalence)
        # Check images copy
        all_records_cameras = list(kapture.flatten(kapture_data.records_camera))
        for _, _, name in all_records_cameras:
            image_path = get_image_fullpath(self._temp_dirpath, name)
            self.assertTrue(path.isfile(image_path), f"image link {image_path}")

    def test_maupertuis_export_db_only(self):
        # export/import and check
        colmap_db_filepath = path.join(self._temp_dirpath, 'colmap.db')
        export_colmap(self._kapture_dirpath, colmap_db_filepath, None,
                      self.keypoints_type, self.descriptors_type, None, True)

        kapture_data = import_colmap(self._temp_dirpath, colmap_db_filepath,
                                     keypoints_type=self.keypoints_type,
                                     descriptors_type=self.descriptors_type,
                                     force_overwrite_existing=True,
                                     no_geometric_filtering=True)

        # check the numbers
        self.assertEqual(1, len(kapture_data.sensors))
        self.assertEqual(4, len(kapture_data.trajectories))
        self.assertEqual(4, len(kapture_data.records_camera))
        self.assertIs(kapture_data.records_lidar, None)
        self.assertIs(kapture_data.records_wifi, None)
        self.assertEqual(4, len(kapture_data.keypoints[self.keypoints_type]))
        self.assertEqual(4, len(kapture_data.descriptors[self.descriptors_type]))
        self.assertEqual(6, len(kapture_data.matches[self.keypoints_type]))
        self.assertIs(kapture_data.points3d, None)
        self.assertIs(kapture_data.observations, None)

    def test_maupertuis_export(self):
        # export/import and check
        colmap_db_filepath = path.join(self._temp_dirpath, 'colmap.db')
        colmap_txt_filepath = path.join(self._temp_dirpath, 'dense')
        export_colmap(self._kapture_dirpath, colmap_db_filepath, colmap_txt_filepath,
                      self.keypoints_type,
                      self.descriptors_type,
                      None, True)

        kapture_data = import_colmap(self._temp_dirpath, colmap_db_filepath, colmap_txt_filepath,
                                     keypoints_type=self.keypoints_type,
                                     descriptors_type=self.descriptors_type,
                                     force_overwrite_existing=True,
                                     no_geometric_filtering=True)

        # check the numbers
        self.assertEqual(1, len(kapture_data.sensors))
        self.assertEqual(4, len(kapture_data.trajectories))
        self.assertEqual(4, len(kapture_data.records_camera))
        self.assertIs(kapture_data.records_lidar, None)
        self.assertIs(kapture_data.records_wifi, None)
        self.assertIs(kapture_data.records_gnss, None)
        self.assertEqual(4, len(kapture_data.keypoints[self.keypoints_type]))
        self.assertEqual(4, len(kapture_data.descriptors[self.descriptors_type]))
        self.assertEqual(6, len(kapture_data.matches[self.keypoints_type]))
        self.assertEqual(1039, len(kapture_data.points3d))
        self.assertEqual(1039, len(kapture_data.observations))

        # compare against golden kapture
        kapture_data_golden = kapture_from_dir(self._kapture_dirpath)
        # drop GPS, Wifi, Lidar
        kapture_data.records_lidar = None
        kapture_data.records_wifi = None
        kapture_data_golden.records_gnss = None
        kapture_data_golden.sensors = kapture.Sensors({
            sensor_id: sensor
            for sensor_id, sensor in kapture_data_golden.sensors.items()
            if isinstance(sensor, kapture.Camera)
        })

        # compare
        equivalence = kapture.algo.compare.equal_kapture(kapture_data, kapture_data_golden)
        self.assertTrue(equivalence)


class TestColmapRig(unittest.TestCase):
    def test_export(self):
        # the rig
        rigs = kapture.Rigs()
        rigs['rig0', get_camera_kapture_id_from_colmap_id(0)] = kapture.PoseTransform()
        rigs['rig0', get_camera_kapture_id_from_colmap_id(1)] = kapture.PoseTransform()
        # the records
        records_camera = kapture.RecordsCamera()
        records_camera[0, get_camera_kapture_id_from_colmap_id(0)] = 'camL/0000.jpg'
        records_camera[0, get_camera_kapture_id_from_colmap_id(1)] = 'camR/0000.jpg'
        records_camera[1, get_camera_kapture_id_from_colmap_id(0)] = 'camL/0001.jpg'
        records_camera[1, get_camera_kapture_id_from_colmap_id(1)] = 'camR/0001.jpg'
        # expect
        expected_rigs = [{
            "cameras": [
                {
                    "camera_id": 0,
                    "image_prefix": "camL"
                },
                {
                    "camera_id": 1,
                    "image_prefix": "camR"
                }
            ],
            "ref_camera_id": 0
        }]

        colmap_camera_ids = {get_camera_kapture_id_from_colmap_id(i): i for i in range(2)}
        colmap_rigs = export_colmap_rig_json(rigs, records_camera, colmap_camera_ids)
        self.assertEqual(colmap_rigs, expected_rigs)

    def test_import(self):
        colmap_rigs = [{
            "cameras": [
                {
                    "camera_id": 0,
                    "image_prefix": "camL"
                },
                {
                    "camera_id": 1,
                    "image_prefix": "camR"
                }
            ],
            "ref_camera_id": 0
        }]

        rigs_kapture, reconstructed_images, reconstructed_trajectories = import_colmap_rig_json(rigs_colmap=colmap_rigs)
        self.assertEqual([('rig0', get_camera_kapture_id_from_colmap_id(0)),
                          ('rig0', get_camera_kapture_id_from_colmap_id(1))],
                         rigs_kapture.key_pairs())
        self.assertIsNone(reconstructed_images)
        self.assertIsNone(reconstructed_trajectories)

        # the records
        images = kapture.RecordsCamera()
        images[0, get_camera_kapture_id_from_colmap_id(0)] = 'camL/0000.jpg'
        images[1, get_camera_kapture_id_from_colmap_id(1)] = 'camR/0000.jpg'
        images[2, get_camera_kapture_id_from_colmap_id(0)] = 'camL/0001.jpg'
        images[3, get_camera_kapture_id_from_colmap_id(1)] = 'camR/0001.jpg'
        rigs_kapture, reconstructed_images, reconstructed_trajectories = import_colmap_rig_json(
            rigs_colmap=colmap_rigs, images=images)
        # check timestamps has been recovered.
        self.assertEqual([(0, get_camera_kapture_id_from_colmap_id(0)),
                          (0, get_camera_kapture_id_from_colmap_id(1)),
                          (1, get_camera_kapture_id_from_colmap_id(0)),
                          (1, get_camera_kapture_id_from_colmap_id(1))],
                         reconstructed_images.key_pairs())

        # trajectories
        trajectories = kapture.Trajectories()
        trajectories[0, get_camera_kapture_id_from_colmap_id(0)] = kapture.PoseTransform()
        trajectories[1, get_camera_kapture_id_from_colmap_id(1)] = kapture.PoseTransform()
        trajectories[2, get_camera_kapture_id_from_colmap_id(0)] = kapture.PoseTransform()
        trajectories[3, get_camera_kapture_id_from_colmap_id(1)] = kapture.PoseTransform()
        rigs_kapture, reconstructed_images, reconstructed_trajectories = import_colmap_rig_json(
            rigs_colmap=colmap_rigs, images=images, trajectories=trajectories)
        self.assertEqual([(0, get_camera_kapture_id_from_colmap_id(0)),
                          (0, get_camera_kapture_id_from_colmap_id(1)),
                          (1, get_camera_kapture_id_from_colmap_id(0)),
                          (1, get_camera_kapture_id_from_colmap_id(1))],
                         reconstructed_trajectories.key_pairs())


if __name__ == '__main__':
    unittest.main()
