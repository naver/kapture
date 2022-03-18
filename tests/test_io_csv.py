#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import numpy as np
import os
import os.path as path
import tempfile
import warnings
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
import kapture.io.csv as csv
from kapture.io.csv import kapture_linesep
import kapture.io.features
import kapture.algo.compare
from kapture.utils.paths import path_secure


########################################################################################################################
# Pose #################################################################################################################
class TestCsvPose(unittest.TestCase):
    def test_pose_write(self):
        pose = kapture.PoseTransform(r=[1.0, 0.0, 0.0, 0.0], t=[0.0, 0.0, 0.0])
        fields = csv.pose_to_list(pose)
        self.assertListEqual(fields, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pose = kapture.PoseTransform(r=[0.5, 0.5, 0.5, 0.5], t=[0., 0., 0.])
        fields = csv.pose_to_list(pose)
        self.assertListEqual(fields, [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
        pose = kapture.PoseTransform(r=[0.5, 0.5, 0.5, 0.5], t=[4., 2., -2.])
        fields = csv.pose_to_list(pose)
        self.assertListEqual(fields, [0.5, 0.5, 0.5, 0.5, 4., 2., -2.])


########################################################################################################################
# File read/write ######################################################################################################
class TestCsvFile(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._temp_filepath = path.join(self._tempdir.name, 'timestamps_data.txt')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_last_line_empty(self):
        with open(self._temp_filepath, 'wt') as fw:
            fw.write('')

        with open(self._temp_filepath, 'r') as fr:
            line = csv.get_last_line(fr)
        self.assertEqual(line, '')

    def test_last_line_small_file(self):
        content = kapture_linesep.join([csv.KAPTURE_FORMAT_1,
                                        '# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz',
                                        '0001, cam0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0',
                                        '0002, cam1,  0.5,  0.5,  0.5,  0.5,  4.0,  2.0, -2.0'
                                        ])
        last_line = '1000, cam2,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0'
        with open(self._temp_filepath, 'wt') as fw:
            fw.write(content)
            fw.write(kapture_linesep + last_line)

        with open(self._temp_filepath, 'r') as fr:
            line = csv.get_last_line(fr)
        self.assertEqual(line, last_line)

    def test_last_line_big_file(self):
        content = kapture_linesep.join([csv.KAPTURE_FORMAT_1,
                                        '# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz',
                                        '0001, cam0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0'
                                        ])
        some_line = '0002, cam1,  0.5,  0.5,  0.5,  0.5,  4.0,  2.0, -2.0'
        last_line = '1000, cam2,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0'
        # Generate a big file
        with open(self._temp_filepath, 'wt') as fw:
            fw.write(content)
            for i in range(0, 100000):
                fw.write(kapture_linesep + some_line)
            fw.write(kapture_linesep + last_line)
        with open(self._temp_filepath, 'r') as fr:
            line = csv.get_last_line(fr)
        self.assertEqual(line, last_line)


########################################################################################################################
# Sensors ##############################################################################################################
class TestCsvSensors(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._temp_filepath = path.join(self._tempdir.name, 'sensors.txt')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_sensor_write(self):
        cam0 = kapture.Camera(name='the_name', camera_type='SIMPLE_PINHOLE', camera_params=[640, 480, 100, 320, 240])
        sensor_fields = csv.sensor_to_list(cam0)
        self.assertIsInstance(sensor_fields, list)
        self.assertEqual(len(sensor_fields), 8)
        self.assertEqual(sensor_fields,
                         ['the_name', kapture.SensorType.camera.name,
                          'SIMPLE_PINHOLE', '640', '480', '100', '320', '240'])

    def test_sensor_file_version(self):
        cam0 = kapture.Camera(name='cam0', camera_type='SIMPLE_PINHOLE', camera_params=[640, 480, 100, 320, 240])
        sensors = kapture.Sensors()
        sensors['cam0'] = cam0
        csv.sensors_to_file(self._temp_filepath, sensors)
        version = csv.get_version_from_csv_file(self._temp_filepath)
        current_version = csv.current_format_version()
        self.assertEqual(current_version, version, "Version correctly stored")

    def test_sensors_write(self):
        cam0 = kapture.Camera(name='cam0', camera_type='SIMPLE_PINHOLE', camera_params=[640, 480, 100, 320, 240])
        cam1 = kapture.Camera(name='cam1', camera_type='SIMPLE_PINHOLE', camera_params=[640, 480, 100, 320, 240])
        formatted_expected = kapture_linesep.join([csv.KAPTURE_FORMAT_1,
                                                   '# sensor_id, name, sensor_type, [sensor_params]+',
                                                   'cam0, cam0, camera, SIMPLE_PINHOLE, 640, 480, 100, 320, 240',
                                                   'cam1, cam1, camera, SIMPLE_PINHOLE, 640, 480, 100, 320, 240',
                                                   ''])
        sensors = kapture.Sensors()
        sensors['cam0'] = cam0
        sensors['cam1'] = cam1
        csv.sensors_to_file(self._temp_filepath, sensors)
        with open(self._temp_filepath, 'rt') as f:
            formatted_actual = ''.join(f.readlines())

        self.assertEqual(formatted_actual, formatted_expected)

    def test_sensors_read(self):
        formatted_expected = kapture_linesep.join([
            '# sensor_id, name, sensor_type, [sensor_params]+',
            'cam0, cam0, camera, SIMPLE_PINHOLE, 640, 480, 100, 320, 240',
            'cam1, cam1, camera, SIMPLE_PINHOLE, 640, 480, 100, 320, 240',
            ''])

        with open(self._temp_filepath, 'wt') as f:
            f.write(formatted_expected)

        sensors = csv.sensors_from_file(self._temp_filepath)
        self.assertIsInstance(sensors, kapture.Sensors)
        self.assertEqual(len(sensors), 2)
        self.assertIn('cam0', sensors)
        self.assertIn('cam1', sensors)
        self.assertEqual('cam0', sensors['cam0'].name)
        self.assertEqual('cam1', sensors['cam1'].name)
        self.assertEqual(kapture.SensorType.camera.name, sensors['cam0'].sensor_type)
        self.assertEqual(kapture.SensorType.camera.name, sensors['cam1'].sensor_type)
        self.assertEqual(6, len(sensors['cam1'].sensor_params))
        self.assertListEqual(sensors['cam0'].sensor_params, ['SIMPLE_PINHOLE', '640', '480', '100', '320', '240'])


########################################################################################################################
# Rigs #################################################################################################################
class TestCsvRigs(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._temp_filepath = path.join(self._tempdir.name, 'rigs.txt')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_rig_write(self):
        rigs = kapture.Rigs()
        rigs['rig1', 'cam0'] = kapture.PoseTransform()
        rigs['rig1', 'cam1'] = kapture.PoseTransform(r=[0.5, 0.5, 0.5, 0.5])
        content_expected = kapture_linesep.join([csv.KAPTURE_FORMAT_1,
                                                 '# rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz',
                                                 'rig1, cam0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0',
                                                 'rig1, cam1,  0.5,  0.5,  0.5,  0.5,  0.0,  0.0,  0.0',
                                                 ''])

        csv.rigs_to_file(self._temp_filepath, rigs)
        with open(self._temp_filepath, 'rt') as f:
            content_actual = ''.join(f.readlines())
        self.assertEqual(content_expected, content_actual)


########################################################################################################################
# Trajectories #########################################################################################################
class TestCsvTrajectories(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._temp_filepath = path.join(self._tempdir.name, 'trajectories.txt')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_trajectories_write(self):
        pose1 = kapture.PoseTransform(r=[1.0, 0.0, 0.0, 0.0], t=[0.0, 0.0, 0.0])
        pose2 = kapture.PoseTransform(r=[0.5, 0.5, 0.5, 0.5], t=[4., 2., -2.])
        content_expected = [csv.KAPTURE_FORMAT_1 + kapture_linesep,
                            '# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz' + kapture_linesep,
                            '       0, cam0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0' + kapture_linesep,
                            '       0, cam1,  0.5,  0.5,  0.5,  0.5,  4.0,  2.0, -2.0' + kapture_linesep,
                            '     100, cam2,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0' + kapture_linesep
                            ]
        trajectories = kapture.Trajectories()
        timestamp1, timestamp2 = 0, 100
        sensor_id1, sensor_id2, sensor_id3 = 'cam0', 'cam1', 'cam2'
        trajectories[(timestamp1, sensor_id1)] = pose1
        trajectories[(timestamp1, sensor_id2)] = pose2
        trajectories[(timestamp2, sensor_id3)] = pose1

        csv.trajectories_to_file(self._temp_filepath, trajectories)
        with open(self._temp_filepath, 'rt') as f:
            content_actual = f.readlines()

        self.assertListEqual(content_actual, content_expected)

    def test_trajectories_read(self):
        content = [
            '# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz',
            '0, cam0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0',
            '    0, cam1, 0.5, 0.5, 0.5, 0.5, 4.0, 2.0, -2.0',
            '     100,  cam2,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0'
        ]

        with open(self._temp_filepath, 'wt') as f:
            f.write(kapture_linesep.join(content))
        device_ids = {'cam0', 'cam1', 'cam2'}
        trajectories = csv.trajectories_from_file(self._temp_filepath, device_ids)
        self.assertIsInstance(trajectories, kapture.Trajectories)
        self.assertEqual(2, len(trajectories.keys()))  # timestamps
        self.assertEqual(3, len(trajectories.key_pairs()))  # timestamp x devices
        self.assertIn(0, trajectories)
        self.assertIn(100, trajectories)
        self.assertIn('cam0', trajectories[0])
        self.assertIn('cam1', trajectories[0])

        pose = kapture.PoseTransform(r=[0.5, 0.5, 0.5, 0.5], t=[4., 2., -2.])
        self.assertAlmostEqual(trajectories[(0, 'cam1')].r_raw, pose.r_raw)
        self.assertAlmostEqual(trajectories[(0, 'cam1')].t_raw, pose.t_raw)


########################################################################################################################
# Gnss ##############################################################################################################
class TestCsvGnss(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._sensors_filepath = path.join(self._tempdir.name, 'sensors.txt')
        sensors_content = [
            '# sensor_id, name, sensor_type, [sensor_params]+',
            'gps1, gps_01, gnss, EPSG:4326',
            'gps2, gps_02, gnss, EPSG:4326',
        ]
        with open(self._sensors_filepath, 'wt') as f:
            f.write(kapture_linesep.join(sensors_content))

        self._gnss_filepath = path.join(self._tempdir.name, 'records_gnss.txt')
        gnss_content = [
            '# timestamp, device_id, x, y, z, utc, dop',
            '0, gps1, 28.099134, 49.38892, 8.0, 514850398, 3.0',
            '  1, gps1, 29.099134, 50.38892, 9.0, 514850399, 2.0',
            '  1, gps2, 29.099134, 50.38892, 9.0, 514850399, 2.0',
            '     2,   gps1, 30.099134, 51.38892, 10.0, 514850400, 1.0'
        ]
        with open(self._gnss_filepath, 'wt') as f:
            f.write(kapture_linesep.join(gnss_content))

    def tearDown(self):
        self._tempdir.cleanup()

    def test_records_gnss_read(self):
        sensors = csv.sensors_from_file(self._sensors_filepath)
        epsg_codes = {sensor_id: sensor.sensor_params[0]
                      for sensor_id, sensor in sensors.items()
                      if sensor.sensor_type == kapture.SensorType.gnss.name}
        self.assertDictEqual({'gps1': 'EPSG:4326', 'gps2': 'EPSG:4326'}, epsg_codes)

        records_gnss = csv.records_gnss_from_file(self._gnss_filepath)
        self.assertEqual(3, len(records_gnss))
        self.assertEqual(4, len(sorted(kapture.flatten(records_gnss))))
        self.assertIn((2, 'gps1'), records_gnss)
        self.assertEqual(51.388920, records_gnss[2, 'gps1'].y)
        self.assertEqual(30.099134, records_gnss[2, 'gps1'].x)
        self.assertEqual(10.000000, records_gnss[2, 'gps1'].z)
        self.assertEqual(514850400, records_gnss[2, 'gps1'].utc)
        self.assertEqual(1.0, records_gnss[2, 'gps1'].dop)

    def test_records_gnss_read_write_read(self):
        sensors = csv.sensors_from_file(self._sensors_filepath)
        records_gnss = csv.records_gnss_from_file(self._gnss_filepath)

        second_sensors_filepath = self._sensors_filepath + '.cpy'
        second_gnss_filepath = self._gnss_filepath + '.cpy'
        csv.sensors_to_file(second_sensors_filepath, sensors)
        csv.records_gnss_to_file(second_gnss_filepath, records_gnss)
        second_sensors = csv.sensors_from_file(second_sensors_filepath)
        second_records_gnss = csv.records_gnss_from_file(second_gnss_filepath)
        self.assertEqual(len(sensors), len(second_sensors))
        # self.assertEqual(sensors['gps1'], second_sensors['gps1'])
        self.assertEqual(len(records_gnss), len(second_records_gnss))


########################################################################################################################
# Records ##############################################################################################################
class TestCsvRecords(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_records_read(self):
        pass


########################################################################################################################
# keypoints ############################################################################################################
class TestCsvKeypoints(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_dirpath = self._tempdir.name
        self._keypoints_dirpath = path.join(self._kapture_dirpath, 'reconstruction', 'keypoints')
        self._keypoints_type = 'SIFT'

    def tearDown(self):
        self._tempdir.cleanup()

    def test_keypoints_read_from_files(self):
        images_ids = set(f'cam{cam}/{timestamp:05d}.jpg'
                         for cam in range(2)
                         for timestamp in range(2))
        # make up keypoints files
        keypoints_config_filepath = path.join(self._keypoints_dirpath, self._keypoints_type, 'keypoints.txt')
        os.makedirs(path.dirname(keypoints_config_filepath), exist_ok=True)
        with open(keypoints_config_filepath, 'wt') as f:
            f.write('SIFT, float, 4')
        keypoints_fullpaths = [
            path_secure(path.join(self._keypoints_dirpath, self._keypoints_type, image_id + '.kpt'))
            for image_id in images_ids
        ]
        for keypoints_fullpath in keypoints_fullpaths:
            os.makedirs(path.dirname(keypoints_fullpath), exist_ok=True)
            with open(keypoints_fullpath, 'wt') as f:
                f.write(' ')

        # lock and load
        keypoints = csv.keypoints_from_dir(self._keypoints_type, self._kapture_dirpath, None)

        # check
        self.assertEqual('SIFT', keypoints.type_name)
        self.assertEqual(4, len(keypoints))
        keypoints_filepaths = kapture.io.features.keypoints_to_filepaths(keypoints,
                                                                         self._keypoints_type,
                                                                         self._kapture_dirpath)
        image_filenames_expected = {path_secure(os.path.join(f'cam{ci}', f'{ts:05d}.jpg'))
                                    for ci in [0, 1] for ts in [0, 1]}
        feature_filepaths_expected = {
            path_secure(os.path.join(f'{self._kapture_dirpath}', 'reconstruction',
                                     'keypoints', self._keypoints_type, f'cam{ci}', f'{ts:05d}.jpg.kpt'))
            for ci in [0, 1] for ts in [0, 1]}

        self.assertEqual(image_filenames_expected, set(keypoints_filepaths.keys()))
        self.assertEqual(feature_filepaths_expected, set(keypoints_filepaths.values()))

    def test_keypoints_read_from_images(self):
        # Create
        images_ids = set(f'cam{cam}/{timestamp:05d}.jpg'
                         for cam in range(2)
                         for timestamp in range(2))
        keypoints_config_filepath = path.join(self._keypoints_dirpath, self._keypoints_type, 'keypoints.txt')
        os.makedirs(path.dirname(keypoints_config_filepath), exist_ok=True)
        with open(keypoints_config_filepath, 'wt') as f:
            f.write('SIFT, float, 4')

        # lock and load
        keypoints = csv.keypoints_from_dir(self._keypoints_type, self._kapture_dirpath, images_ids)

        # check its empty
        self.assertEqual('SIFT', keypoints.type_name)
        self.assertEqual(0, len(keypoints))

        valid = kapture.io.features.keypoints_check_dir(keypoints, self._keypoints_type, self._kapture_dirpath)
        self.assertTrue(valid)

        # create actual files
        for images_id in images_ids:
            keypoint_filepath = path.join(self._keypoints_dirpath, self._keypoints_type, images_id + '.kpt')
            os.makedirs(path.dirname(keypoint_filepath), exist_ok=True)
            with open(keypoint_filepath, 'wt') as f:
                f.write('')

        # lock and load again
        keypoints = csv.keypoints_from_dir(self._keypoints_type, self._kapture_dirpath, images_ids)
        self.assertEqual('SIFT', keypoints.type_name)
        self.assertEqual(4, len(keypoints))

        keypoints_filepaths = kapture.io.features.keypoints_to_filepaths(keypoints,
                                                                         self._keypoints_type,
                                                                         self._kapture_dirpath)
        image_filenames_expected = {f'cam{ci}/{ts:05d}.jpg'
                                    for ci in [0, 1] for ts in [0, 1]}
        feature_filepaths_expected = {
            path_secure(f'{self._kapture_dirpath}/reconstruction/keypoints'
                        f'/{self._keypoints_type}/cam{ci}/{ts:05d}.jpg.kpt')
            for ci in [0, 1] for ts in [0, 1]}

        self.assertEqual(image_filenames_expected, set(keypoints_filepaths))
        self.assertEqual(feature_filepaths_expected, set(keypoints_filepaths.values()))

        valid = kapture.io.features.keypoints_check_dir(keypoints, self._keypoints_type, self._kapture_dirpath)
        self.assertTrue(valid)

        # destroy files and check
        os.remove(path.join(self._keypoints_dirpath, self._keypoints_type, 'cam0/00000.jpg.kpt'))
        valid = kapture.io.features.keypoints_check_dir(keypoints, self._keypoints_type, self._kapture_dirpath)
        self.assertFalse(valid)


########################################################################################################################
# points3d #############################################################################################################
class TestCsvPoints3d(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tempdir.cleanup()

    def test_points_to_files(self):
        data = np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 255, 0, 0],
            [0, 0, 1, 255, 0, 255],
        ])
        points3d = kapture.Points3d(data)

        filepath = path.join(self._tempdir.name, 'points3d.txt')
        csv.points3d_to_file(filepath, points3d)
        with open(filepath, 'rt') as file:
            lines = file.readlines()

        self.assertEqual(5, len(lines))
        self.assertTrue(lines[0].startswith('#'))
        self.assertTrue(lines[1].startswith('#'))
        first_line = [float(f) for f in lines[2].split(',')]
        self.assertEqual(6, len(first_line))
        self.assertAlmostEqual([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], first_line)

    def test_points_from_empty_file(self):
        filepath = path.join(self._tempdir.name, 'points3d.txt')
        with open(filepath, 'wt') as file:
            file.write(csv.KAPTURE_FORMAT_1 + kapture_linesep)
            file.write('# X, Y, Z, R, G, B' + kapture_linesep)
        # prevent numpy showing warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points3d = csv.points3d_from_file(filepath)
        self.assertEqual(0, points3d.shape[0])

    def test_points_from_single_line_file(self):
        filepath = path.join(self._tempdir.name, 'points3d.txt')
        with open(filepath, 'wt') as file:
            file.write(csv.KAPTURE_FORMAT_1 + kapture_linesep)
            file.write('# X, Y, Z, R, G, B' + kapture_linesep)
            file.write('0, 1.0, 0.0, 255, .0, 150' + kapture_linesep)
        points3d = csv.points3d_from_file(filepath)
        self.assertEqual(1, points3d.shape[0])
        self.assertAlmostEqual([0.0, 1.0, 0.0, 255.0, .0, 150.0], points3d[0])

    def test_points_from_file(self):
        filepath = path.join(self._tempdir.name, 'points3d.txt')
        with open(filepath, 'wt') as file:
            file.write(csv.KAPTURE_FORMAT_1 + kapture_linesep)
            file.write('# X, Y, Z, R, G, B' + kapture_linesep)
            file.write('0, 1.0, 0.0, 255, .0, 150' + kapture_linesep)
            file.write('-10, 0, 0.0, 25, .0, 150' + kapture_linesep)
            file.write('1545, 0, 0.0, 25, .0, 150' + kapture_linesep)
        points3d = csv.points3d_from_file(filepath)
        self.assertEqual(3, points3d.shape[0])
        self.assertAlmostEqual([0.0, 1.0, 0.0, 255.0, .0, 150.0], points3d[0])
        self.assertAlmostEqual([1545.0, 0.0, 0.0, 25.0, .0, 150.0], points3d[2])


########################################################################################################################
# Observations #########################################################################################################
class TestCsvObservations(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._observations_expected_filepath = path.join(self._tempdir.name, 'expected', 'observations.txt')
        self._observations_actual_filepath = path.join(self._tempdir.name, 'actual', 'observations.txt')
        # creates ground truth couple data/file
        self._observations_expected = kapture.Observations({
            0: {'SIFT': [('image1.jpg', 0), ('image2.jpg', 0)]},
            2: {'SIFT': [('image1.jpg', 2), ('image2.jpg', 3)]}
        })
        self._observations_csv_expected = kapture_linesep.join([csv.KAPTURE_FORMAT_1,
                                                                "# point3d_id, keypoints_type,"
                                                                " [image_path, feature_id]*",
                                                                "0, SIFT, image1.jpg, 0, image2.jpg, 0",
                                                                "2, SIFT, image1.jpg, 2, image2.jpg, 3"])
        self._observations_csv_expected += kapture_linesep
        os.makedirs(path.dirname(self._observations_expected_filepath), exist_ok=True)
        with open(self._observations_expected_filepath, 'wt') as file:
            file.write(self._observations_csv_expected)

    def tearDown(self):
        self._tempdir.cleanup()

    def test_observations_to_file(self):
        csv.observations_to_file(self._observations_actual_filepath, self._observations_expected)
        with open(self._observations_actual_filepath, 'rt') as file:
            lines = file.readlines()
            content_actual = ''.join(lines)
        self.assertEqual(self._observations_csv_expected, content_actual)

    def test_observations_from_file(self):
        observations_actual = csv.observations_from_file(self._observations_expected_filepath)
        self.assertEqual(self._observations_expected, observations_actual)

    def test_observations_from_file_limited(self):
        observations_actual = csv.observations_from_file(self._observations_expected_filepath, {'SIFT': {'image1.jpg'}})
        self.assertEqual(2, len(observations_actual))
        self.assertNotEqual(self._observations_expected, observations_actual)


########################################################################################################################
# Kapture ##############################################################################################################
class TestCsvKapture(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tempdir.cleanup()

    def test_kapture_write(self):
        kapture_data = kapture.Kapture()

        # test it is not writing files for undefined parts
        csv.kapture_to_dir(self._tempdir.name, kapture_data)
        self.assertFalse(path.exists(path.join(self._tempdir.name, 'sensors', 'sensors.txt')))
        self.assertFalse(path.exists(path.join(self._tempdir.name, 'sensors', 'trajectories.txt')))
        self.assertFalse(path.exists(path.join(self._tempdir.name, 'sensors', 'rigs.txt')))

        # test it is actually writing files for parts
        kapture_data.sensors = kapture.Sensors()
        kapture_data.trajectories = kapture.Trajectories()
        kapture_data.rigs = kapture.Rigs()
        csv.kapture_to_dir(self._tempdir.name, kapture_data)
        self.assertTrue(path.exists(path.join(self._tempdir.name, 'sensors', 'sensors.txt')))
        self.assertTrue(path.exists(path.join(self._tempdir.name, 'sensors', 'trajectories.txt')))
        self.assertTrue(path.exists(path.join(self._tempdir.name, 'sensors', 'rigs.txt')))
        # TODO: using samples

    def test_kapture_write_read(self):
        kapture_data_expected = kapture.Kapture()
        kapture_data_expected.sensors = kapture.Sensors()
        kapture_data_expected.trajectories = kapture.Trajectories()
        kapture_data_expected.rigs = kapture.Rigs()
        csv.kapture_to_dir(self._tempdir.name, kapture_data_expected)
        # TODO: using samples

    def test_kapture_format_version_from_disk(self):
        kapture_data = kapture.Kapture()
        kapture_data.sensors = kapture.Sensors()
        csv.kapture_to_dir(self._tempdir.name, kapture_data)
        version = csv.kapture_format_version(self._tempdir.name)
        self.assertEqual(csv.current_format_version(), version, "We have the current version")

    def test_kapture_format_version_memory(self):
        kapture_data = kapture.Kapture()
        kapture_data.sensors = kapture.Sensors()
        csv.kapture_to_dir(self._tempdir.name, kapture_data)
        kapture_read = csv.kapture_from_dir(self._tempdir.name)
        self.assertEqual(csv.current_format_version(), kapture_read.format_version, "We have the current version")


########################################################################################################################
# M1X ##################################################################################################################
class TestCsvM1x(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._samples_folder = path.abspath(path.join(path.dirname(__file__), '..', 'samples', 'm1x'))
        self._kapture_data = csv.kapture_from_dir(self._samples_folder)

    def tearDown(self):
        self._tempdir.cleanup()

    def test_sensors_read_file(self):
        self.assertIsInstance(self._kapture_data.sensors, kapture.Sensors)
        self.assertEqual(len(self._kapture_data.sensors), 14)
        self.assertIn('lidar0', self._kapture_data.sensors)
        self.assertIn('22970291', self._kapture_data.sensors)
        self.assertIn('AC01324969', self._kapture_data.sensors)
        self.assertIn('AC01324954_wifi', self._kapture_data.sensors)
        self.assertIn('AC01324954_bluetooth', self._kapture_data.sensors)
        self.assertEqual('horizontal', self._kapture_data.sensors['lidar1'].name)
        self.assertEqual(kapture.SensorType.camera.name, self._kapture_data.sensors['22970291'].sensor_type)
        self.assertEqual(kapture.SensorType.lidar.name, self._kapture_data.sensors['lidar0'].sensor_type)
        self.assertEqual(15, len(self._kapture_data.sensors['22970291'].sensor_params))

    def test_rigs_read_file(self):
        self.assertIsInstance(self._kapture_data.rigs, kapture.Rigs)
        # there is a single rig called "rig"
        self.assertEqual(len(self._kapture_data.rigs.keys()), 1)
        self.assertIn('rig', self._kapture_data.rigs)
        # there is 13 sensors in this rig
        self.assertEqual(len(self._kapture_data.rigs['rig']), 14)
        self.assertIn('lidar0', self._kapture_data.rigs['rig'])

    def test_trajectories_read_file(self):
        self.assertIsInstance(self._kapture_data.trajectories, kapture.Trajectories)
        # there is 9 timestamps
        self.assertEqual(len(self._kapture_data.trajectories.keys()), 9)
        # there is also 9 timestamps+device
        self.assertEqual(len(self._kapture_data.trajectories.key_pairs()), 9)

    def test_records_camera_read_file(self):
        self.assertEqual(8, len(self._kapture_data.records_camera.keys()))
        self.assertEqual(18, len(self._kapture_data.records_camera.key_pairs()))
        self.assertIn(1555399760151000, self._kapture_data.records_camera)
        self.assertIn(1555399761779869, self._kapture_data.records_camera)
        self.assertIn('AC01324954', self._kapture_data.records_camera[1555399760151000])
        self.assertEqual('AC01324954/AC01324954_1555399760151000.jpg',
                         self._kapture_data.records_camera[1555399760151000]['AC01324954'])

    def test_records_wifi_read_file(self):
        self.assertEqual(2, len(self._kapture_data.records_wifi.keys()))
        self.assertEqual(2, len(self._kapture_data.records_wifi.key_pairs()))
        self.assertIn(1555398770307, self._kapture_data.records_wifi)
        self.assertIn('AC01324954_wifi', self._kapture_data.records_wifi[1555398770307])
        record_wifi_expected = kapture.RecordWifi({
            '68:72:51:80:52:df': kapture.RecordWifiSignal(frequency=2417, rssi=-33.0, ssid='M1X_PicoM2'),
            '68:9c:e2:e1:b0:60': kapture.RecordWifiSignal(frequency=5765, rssi=-49, ssid='@HYUNDAI-WiFi')
        })
        # compare representation, to be robust to ?????
        self.assertEqual(str(record_wifi_expected),
                         str(self._kapture_data.records_wifi[1555398770307, 'AC01324954_wifi']))
        record_wifi_expected = kapture.RecordWifi({
            '68:72:51:80:52:df': kapture.RecordWifiSignal(frequency=2417, rssi=-35, ssid='M1X_PicoM2'),
            '68:9c:e2:e1:b0:60': kapture.RecordWifiSignal(frequency=5765, rssi=-47, ssid='@HYUNDAI-WiFi')
        })
        self.assertEqual(str(record_wifi_expected),
                         str(self._kapture_data.records_wifi[1555398771307, 'AC01324954_wifi']))

    def test_records_bluetooth_read_file(self):
        self.assertEqual(2, len(self._kapture_data.records_bluetooth.keys()))
        self.assertEqual(2, len(self._kapture_data.records_bluetooth.key_pairs()))
        self.assertIn(1555398770307, self._kapture_data.records_bluetooth)
        self.assertIn('AC01324954_bluetooth', self._kapture_data.records_bluetooth[1555398770307])
        record_bluetooth_expected = kapture.RecordBluetooth({
            '35:A8:4B:D9:95:06': kapture.RecordBluetoothSignal(rssi=-73.0),
            '6F:80:17:51:5C:16': kapture.RecordBluetoothSignal(rssi=-88, name='MyPhone')
        })
        # compare representation, to be robust to ?????
        self.assertEqual(str(record_bluetooth_expected),
                         str(self._kapture_data.records_bluetooth[1555398770307, 'AC01324954_bluetooth']))
        record_wifi_expected = kapture.RecordWifi({
            '52:BE:1D:75:47:A1': kapture.RecordBluetoothSignal(rssi=-89),
            '94:65:2D:A6:EF:C4': kapture.RecordBluetoothSignal(rssi=-59)
        })
        self.assertEqual(str(record_wifi_expected),
                         str(self._kapture_data.records_bluetooth[1555398771307, 'AC01324954_bluetooth']))

    def test_records_lidar_read_file(self):
        self.assertEqual(3, len(self._kapture_data.records_lidar.keys()))
        self.assertEqual(6, len(self._kapture_data.records_lidar.key_pairs()))
        self.assertEqual(self._kapture_data.records_lidar[1555399760777612, 'lidar0'], 'lidar0/1555399760777612.pcd')
        self.assertEqual(self._kapture_data.records_lidar[1555399760777612, 'lidar1'], 'lidar1/1555399760777612.pcd')
        self.assertEqual(self._kapture_data.records_lidar[1555399760878721, 'lidar0'], 'lidar0/1555399760878721.pcd')
        self.assertEqual(self._kapture_data.records_lidar[1555399760878721, 'lidar1'], 'lidar1/1555399760878721.pcd')
        self.assertEqual(self._kapture_data.records_lidar[1555399760979021, 'lidar0'], 'lidar0/1555399760979021.pcd')
        self.assertEqual(self._kapture_data.records_lidar[1555399760979021, 'lidar1'], 'lidar1/1555399760979021.pcd')

    def test_read_write(self):
        kapture_data = csv.kapture_from_dir(self._samples_folder)
        csv.kapture_to_dir(self._tempdir.name, kapture_data)
        # TODO


########################################################################################################################
# maupertuis ###########################################################################################################
class TestCsvMaupertuis(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._samples_folder = path.abspath(path.join(path.dirname(__file__), '..', 'samples', 'maupertuis'))
        self._kapture_dirpath = path.join(self._samples_folder, 'kapture')
        self._features_type = 'SIFT'
        self._kapture_data = csv.kapture_from_dir(self._kapture_dirpath)

    def tearDown(self):
        self._tempdir.cleanup()

    def test_keypoints_from_dir(self):
        keypoints = kapture.io.csv.keypoints_from_dir(self._features_type, self._kapture_dirpath, None)
        self.assertEqual(keypoints.dtype, np.float32)
        self.assertEqual(keypoints.dsize, 6)
        self.assertEqual(set(keypoints), {'03.jpg', '01.jpg', '02.jpg', '00.jpg'})

    def test_descriptors_from_dir(self):
        descriptors = kapture.io.csv.descriptors_from_dir(self._features_type, self._kapture_dirpath, None, None)
        self.assertEqual(descriptors.dtype, np.uint8)
        self.assertEqual(descriptors.dsize, 128)
        self.assertEqual(set(descriptors), {'03.jpg', '01.jpg', '02.jpg', '00.jpg'})

    def test_matches_from_dir(self):
        matches = kapture.io.csv.matches_from_dir(self._features_type, self._kapture_dirpath)
        self.assertEqual({('00.jpg', '03.jpg'), ('01.jpg', '03.jpg'),
                          ('00.jpg', '02.jpg'), ('02.jpg', '03.jpg'),
                          ('01.jpg', '02.jpg'), ('00.jpg', '01.jpg')},
                         set(matches))

    def test_observations_from_file(self):
        image_filenames_with_keypoints = {self._features_type: {'03.jpg', '01.jpg', '02.jpg', '00.jpg'}}
        observations_filepath = kapture.io.csv.get_csv_fullpath(kapture.Observations, self._kapture_dirpath)
        observations = kapture.io.csv.observations_from_file(observations_filepath, image_filenames_with_keypoints)
        self.assertEqual(1039, len(observations))
        # observations[point3d_idx, keypoints_type]= [(image_filename, keypoint_id), (image_filename, keypoint_id), ...]
        observation0 = observations[0, self._features_type]
        self.assertEqual(
            [('01.jpg', 4561), ('02.jpg', 3389), ('00.jpg', 4975), ('03.jpg', 3472)],
            observation0
        )

    def test_overall_read(self):
        # check the overall loading went fine
        self.assertEqual(2, len(self._kapture_data.sensors))
        self.assertEqual(4, len(self._kapture_data.trajectories))
        self.assertEqual(4, len(self._kapture_data.records_camera))
        self.assertEqual(4, len(self._kapture_data.records_gnss))
        self.assertEqual(4, len(self._kapture_data.keypoints[self._features_type]))
        self.assertEqual(4, len(self._kapture_data.descriptors[self._features_type]))
        self.assertEqual(6, len(self._kapture_data.matches[self._features_type]))
        self.assertEqual(1039, len(self._kapture_data.observations))
        self.assertEqual(1039, len(self._kapture_data.points3d))


if __name__ == '__main__':
    unittest.main()
