#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the extraction of periods of a kapture (kapture.algo.extract, kapture_extract_period).
"""

import json
import os
import os.path as path
import tempfile
import unittest

import numpy as np

import kapture

from kapture.algo.extract import extract_period, kapture_extract_period, normalize_windows, time_range
from kapture.algo.extract import box_area, box_bounds, kapture_extract_area, points_in_box, points_in_sphere
from kapture.algo.extract import sphere_area, trajectory_positions, windows_in_area
from kapture.io.csv import kapture_from_dir, kapture_to_dir
from kapture.io.records import TransferAction, get_record_fullpath

T_START = 1_000_000_000  # microseconds
STEP = 100_000  # 10 Hz
NB_FRAMES = 20


def make_kapture() -> kapture.Kapture:
    """ 2 cameras + lidar (derived lidar0_clean) on a rig, IMU, wifi, 20 frames at 10 Hz; no pose for frame 0 """
    sensors = kapture.Sensors()
    for camera_id in ['cam0', 'cam1']:
        sensors[camera_id] = kapture.Camera(kapture.CameraType.SIMPLE_PINHOLE, [640, 480, 500, 320, 240])
    for sensor_id, sensor_type in [('lidar0', 'lidar'), ('lidar0_clean', 'lidar'), ('gyro', 'gyroscope'),
                                   ('wifi', 'wifi')]:
        sensors[sensor_id] = kapture.create_sensor(sensor_type, [])
    rigs = kapture.Rigs()
    for sensor_id in ['cam0', 'cam1', 'lidar0', 'lidar0_clean']:
        rigs['rig', sensor_id] = kapture.PoseTransform()
    trajectories = kapture.Trajectories()
    records_camera = kapture.RecordsCamera()
    records_lidar = kapture.RecordsLidar()
    records_gyroscope = kapture.RecordsGyroscope()
    records_wifi = kapture.RecordsWifi()
    for frame in range(NB_FRAMES):
        timestamp = T_START + frame * STEP
        if frame > 0:
            trajectories[timestamp, 'rig'] = kapture.PoseTransform(t=[frame, 0, 0])
        for camera_id in ['cam0', 'cam1']:
            records_camera[timestamp, camera_id] = f'{camera_id}/{timestamp}.jpg'
        for lidar_id in ['lidar0', 'lidar0_clean']:
            records_lidar[timestamp, lidar_id] = f'{lidar_id}/{timestamp}.ply'
        for sub in range(4):
            records_gyroscope[timestamp + sub * STEP // 4, 'gyro'] = kapture.RecordGyroscope(0.1, 0.2, 0.3)
        if frame % 5 == 0:
            records_wifi[timestamp, 'wifi'] = kapture.RecordWifi(
                {'68:72:51:80:52:df': kapture.RecordWifiSignal(frequency=2417, rssi=-33.0, ssid='ap')})
    return kapture.Kapture(sensors=sensors, rigs=rigs, trajectories=trajectories, records_camera=records_camera,
                           records_lidar=records_lidar, records_gyroscope=records_gyroscope, records_wifi=records_wifi)


def write_kapture(kapture_dirpath: str, kapture_data: kapture.Kapture, missing=()) -> None:
    kapture_to_dir(kapture_dirpath, kapture_data)
    for records in [kapture_data.records_camera, kapture_data.records_lidar]:
        for _, _, filename in kapture.flatten(records):
            if filename in missing:
                continue
            filepath = get_record_fullpath(kapture_dirpath, filename)
            os.makedirs(path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as file:
                file.write(filename)
    os.makedirs(path.join(kapture_dirpath, 'sensors', 'raw'))
    with open(path.join(kapture_dirpath, 'sensors', 'raw', 'metadata.json'), 'w') as file:
        file.write('{}')


def frames(*indices) -> list:
    return [T_START + index * STEP for index in indices]


class TestNormalizeWindows(unittest.TestCase):
    def test_absolute_sorted_and_merged(self):
        self.assertEqual(normalize_windows([(50, 60), (10, 20), (15, 30), (30, 35)]), [(10, 35), (50, 60)])

    def test_relative(self):
        kapture_data = make_kapture()
        self.assertEqual(time_range(kapture_data), (T_START, T_START + (NB_FRAMES - 1) * STEP + 3 * STEP // 4))
        self.assertEqual(normalize_windows([(0.5, 1.)], relative=True, kapture_data=kapture_data),
                         [(T_START + 500_000, T_START + 1_000_000)])
        self.assertEqual(normalize_windows([(1, 2)], relative=True, time_unit=1e3, kapture_data=kapture_data),
                         [(T_START + 1000, T_START + 2000)])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            normalize_windows([])
        with self.assertRaises(ValueError):
            normalize_windows([(20, 10)])
        with self.assertRaises(ValueError):
            normalize_windows([(1, 2, 3)])
        with self.assertRaises(ValueError):
            normalize_windows([(0, 1)], relative=True)
        with self.assertRaises(ValueError):
            normalize_windows([(0, 1)], relative=True, kapture_data=kapture.Kapture())

    def test_large_timestamps_are_exact(self):
        self.assertEqual(normalize_windows([(1792656648559471., 1792656685860184.)]),
                         [(1792656648559471, 1792656685860184)])


class TestExtractPeriod(unittest.TestCase):
    def setUp(self):
        self.kapture_data = make_kapture()

    def test_one_window_bounds_included(self):
        extracted = extract_period(self.kapture_data, [(frames(3)[0], frames(6)[0])])
        self.assertEqual(sorted(extracted.records_camera), frames(3, 4, 5, 6))
        self.assertEqual(sorted(extracted.records_lidar), frames(3, 4, 5, 6))
        self.assertEqual(sorted(extracted.trajectories), frames(3, 4, 5, 6))
        self.assertEqual(sorted(extracted.records_wifi), frames(5))
        self.assertEqual(len(extracted.records_gyroscope), 3 * 4 + 1)
        self.assertEqual(extracted.records_camera[frames(3)[0], 'cam1'], f'cam1/{frames(3)[0]}.jpg')
        self.assertEqual(set(extracted.sensors), set(self.kapture_data.sensors))
        self.assertEqual(extracted.rigs, self.kapture_data.rigs)
        self.assertIsInstance(extracted.trajectories, kapture.Trajectories)
        self.assertIsInstance(extracted.records_gyroscope, kapture.RecordsGyroscope)
        self.assertIsNone(extracted.records_depth)
        self.assertIsNone(extracted.points3d)
        # the input is left untouched
        self.assertEqual(len(self.kapture_data.records_camera), NB_FRAMES)

    def test_several_windows(self):
        windows = normalize_windows([(frames(10)[0], frames(11)[0]), (frames(0)[0], frames(1)[0])])
        extracted = extract_period(self.kapture_data, windows)
        self.assertEqual(sorted(extracted.records_camera), frames(0, 1, 10, 11))
        self.assertEqual(sorted(extracted.trajectories), frames(1, 10, 11))  # no pose for frame 0

    def test_between_and_outside(self):
        extracted = extract_period(self.kapture_data, [(frames(3)[0] + 1, frames(4)[0] - 1)])
        self.assertEqual(len(extracted.records_camera), 0)
        self.assertEqual(len(extracted.records_gyroscope), 3)
        extracted = extract_period(self.kapture_data, [(0, 10)])
        self.assertEqual(len(extracted.records_gyroscope), 0)
        self.assertEqual(len(extracted.trajectories), 0)

    def test_skip_sensors(self):
        extracted = extract_period(self.kapture_data, [(frames(3)[0], frames(4)[0])], ['lidar0_clean', 'cam1', 'wifi'])
        self.assertEqual(set(extracted.sensors), {'cam0', 'lidar0', 'gyro'})
        self.assertEqual(set(extracted.rigs['rig']), {'cam0', 'lidar0'})
        self.assertEqual(extracted.records_camera.sensors_ids, {'cam0'})
        self.assertEqual(extracted.records_lidar.sensors_ids, {'lidar0'})
        self.assertEqual(len(extracted.records_wifi), 0)
        # a rig can be skipped too: no rig, no poses
        extracted = extract_period(self.kapture_data, [(frames(3)[0], frames(4)[0])], ['rig'])
        self.assertEqual(len(extracted.rigs), 0)
        self.assertEqual(len(extracted.trajectories), 0)


class TestKaptureExtractPeriod(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.input_path = path.join(self._tempdir.name, 'full')
        self.output_path = path.join(self._tempdir.name, 'sub', 'part')
        self.kapture_data = make_kapture()
        self.window = (frames(3)[0], frames(6)[0])
        self.missing = [f'lidar0_clean/{frames(4)[0]}.ply']
        write_kapture(self.input_path, self.kapture_data, missing=self.missing)

    def tearDown(self):
        self._tempdir.cleanup()

    def _data_files(self, kapture_dirpath: str) -> list:
        data_dirpath = get_record_fullpath(kapture_dirpath)
        return sorted(path.relpath(path.join(dirpath, filename), data_dirpath)
                      for dirpath, _, filenames in os.walk(data_dirpath) for filename in filenames)

    def test_skip_transfer(self):
        kapture_extract_period(self.input_path, self.output_path, [self.window])
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(sorted(reloaded.records_camera), frames(3, 4, 5, 6))
        self.assertEqual(sorted(reloaded.trajectories), frames(3, 4, 5, 6))
        self.assertEqual(len(reloaded.records_gyroscope), 13)
        self.assertEqual(len(list(kapture.flatten(reloaded.records_lidar))), 8)  # nothing dropped: files not checked
        self.assertEqual(self._data_files(self.output_path), [])
        self.assertFalse(path.exists(path.join(self.output_path, 'sensors', 'raw')))
        with open(path.join(self.output_path, 'subset.json')) as file:
            subset = json.load(file)
        self.assertEqual(subset['src'], path.abspath(self.input_path))
        self.assertEqual(subset['windows'], [list(self.window)])
        self.assertEqual(subset['transfer'], 'skip')
        self.assertEqual(subset['records']['records_camera'], 8)
        self.assertEqual(subset['records']['trajectories'], 4)

    def test_copy_drops_the_records_without_file(self):
        kapture_extract_period(self.input_path, self.output_path, [self.window], transfer=TransferAction.copy)
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(len(list(kapture.flatten(reloaded.records_lidar))), 7)
        self.assertNotIn((frames(4)[0], 'lidar0_clean'), reloaded.records_lidar)
        files = self._data_files(self.output_path)
        self.assertEqual(len(files), 8 + 7)
        self.assertEqual(files, sorted(f for _, _, f in kapture.flatten(reloaded.records_camera)) +
                         sorted(f for _, _, f in kapture.flatten(reloaded.records_lidar)))
        filepath = get_record_fullpath(self.output_path, files[0])
        self.assertFalse(path.islink(filepath))
        with open(filepath) as file:
            self.assertEqual(file.read(), files[0])
        self.assertTrue(path.isfile(path.join(self.output_path, 'sensors', 'raw', 'metadata.json')))
        # the input still has everything
        self.assertEqual(len(self._data_files(self.input_path)), 4 * NB_FRAMES - 1)

    def test_links_point_to_the_original_files(self):
        for transfer in [TransferAction.link_relative, TransferAction.link_absolute]:
            with self.subTest(transfer=transfer):
                first_path = path.join(self._tempdir.name, f'first_{transfer.name}')
                second_path = path.join(self._tempdir.name, 'deeper', f'second_{transfer.name}')
                kapture_extract_period(self.input_path, first_path, [self.window], transfer=transfer)
                # extraction of an extraction, relative period
                kapture_extract_period(first_path, second_path, [(0.1, 0.2)], relative=True, transfer=transfer,
                                       skip_sensor_ids=['lidar0_clean'])
                reloaded = kapture_from_dir(second_path)
                self.assertEqual(sorted(reloaded.records_camera), frames(4, 5))
                self.assertNotIn('lidar0_clean', reloaded.sensors)
                files = self._data_files(second_path)
                self.assertEqual(len(files), 4 + 2)
                for filename in files:
                    link = get_record_fullpath(second_path, filename)
                    self.assertTrue(path.islink(link))
                    self.assertEqual(path.isabs(os.readlink(link)), transfer == TransferAction.link_absolute)
                    # one hop to the original file, not a chain through the first extraction
                    target = path.normpath(path.join(path.dirname(link), os.readlink(link)))
                    self.assertEqual(target, path.realpath(get_record_fullpath(self.input_path, filename)))
                raw = path.join(second_path, 'sensors', 'raw', 'metadata.json')
                self.assertTrue(path.islink(raw))
                self.assertEqual(path.realpath(raw),
                                 path.realpath(path.join(self.input_path, 'sensors', 'raw', 'metadata.json')))

    def test_root_link(self):
        kapture_extract_period(self.input_path, self.output_path, [self.window], transfer=TransferAction.root_link)
        self.assertTrue(path.islink(get_record_fullpath(self.output_path).rstrip('/')))
        self.assertTrue(path.isfile(get_record_fullpath(self.output_path, f'cam0/{frames(3)[0]}.jpg')))
        self.assertTrue(path.isfile(path.join(self.output_path, 'sensors', 'raw', 'metadata.json')))

    def test_overwrite_and_refusals(self):
        kapture_extract_period(self.input_path, self.output_path, [self.window], transfer=TransferAction.link_relative)
        kapture_extract_period(self.input_path, self.output_path, [(frames(10)[0], frames(11)[0])],
                               transfer=TransferAction.link_relative, force=True)
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(sorted(reloaded.records_camera), frames(10, 11))
        self.assertEqual(len(self._data_files(self.output_path)), 8)
        with self.assertRaises(ValueError):
            kapture_extract_period(self.input_path, self.output_path, [self.window], transfer=TransferAction.move)
        with self.assertRaises(ValueError):
            kapture_extract_period(self.input_path, self.input_path, [self.window], force=True)
        self.assertEqual(len(kapture_from_dir(self.input_path).records_camera), NB_FRAMES)


def make_walk() -> kapture.Kapture:
    """ make_kapture with a rotated rig walking along x (frame i at x = i, poses for all frames) and 3-D points """
    kapture_data = make_kapture()
    rotation = [0., 0., 0., 1.]  # half turn around z: the pose translation is not the position
    for frame in range(NB_FRAMES):
        pose_world_from_rig = kapture.PoseTransform(r=rotation, t=[frame, 0.5 * (frame % 2), 1.])
        kapture_data.trajectories[T_START + frame * STEP, 'rig'] = pose_world_from_rig.inverse()
    kapture_data.points3d = kapture.Points3d([[x + 0.25, 0.2, z, 255, 0, x] for x in range(-5, 25) for z in [0, 3]])
    return kapture_data


class TestArea(unittest.TestCase):
    def test_points_in_box(self):
        points = np.array([[0.5, 0.5, 0.5], [0, 1, 3], [2, 0.5, 0.5], [2.01, 0.5, 0.5], [0.5, -0.1, 0.5],
                           [0.5, 0.5, 3.5], [0.5, 0.5, -1]])
        box = [0, 2, 0, 1, 0, 3]
        expected = [True, True, True, False, False, False, False]
        self.assertEqual(points_in_box(points, box).tolist(), expected)
        self.assertEqual(box_area(box)(np.hstack([points, np.zeros((len(points), 3))])).tolist(), expected)
        self.assertEqual(points_in_box(np.zeros((0, 3)), box).tolist(), [])
        with self.assertRaises(ValueError):
            box_area([0, 2, 0, 1])
        with self.assertRaises(ValueError):
            box_area([0, 2, 0, 0, 0, 3])
        with self.assertRaises(ValueError):
            box_area([0, 2, 0, 1, 0, -3])

    def test_points_in_box_without_bounds(self):
        points = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1e6], [0.5, 0.5, -1e6], [-7, 0.5, 0.5], [0.5, 9, 0.5],
                           [3, 0.5, 0.5]])
        # no filter on z
        self.assertEqual(points_in_box(points, [0, 2, 0, 1, None, None]).tolist(),
                         [True, True, True, False, False, False])
        # one side only: z >= 0, x <= 2
        self.assertEqual(points_in_box(points, [None, 2, 0, 1, 0, None]).tolist(),
                         [True, True, False, True, False, False])
        self.assertEqual(points_in_box(points, [None] * 6).tolist(), [True] * 6)
        lower, upper = box_bounds([None, 2, 0, 1, 0, None])
        self.assertEqual((lower.tolist(), upper.tolist()), ([-np.inf, 0, 0], [2, 1, np.inf]))
        with self.assertRaises(ValueError):
            box_area([0, 2, 1, 0, None, None])
        with self.assertRaises(ValueError):
            box_area([None] * 4)

    def test_points_in_sphere(self):
        points = np.array([[0, 0, 0], [1, 2, 3], [1, 2, 5], [1, 2, 5.01], [2.5, 3.5, 3], [1, 2, 3, 255, 0, 0][:3]])
        self.assertEqual(points_in_sphere(points, [1, 2, 3], 2).tolist(), [False, True, True, False, False, True])
        self.assertEqual(sphere_area([1, 2, 3], 2)(points).tolist(), [False, True, True, False, False, True])
        self.assertEqual(points_in_sphere(np.zeros((0, 6)), [1, 2, 3], 2).tolist(), [])
        with self.assertRaises(ValueError):
            sphere_area([1, 2], 2)
        with self.assertRaises(ValueError):
            sphere_area([1, 2, 3], 0)

    def test_windows_in_sphere(self):
        kapture_data = make_walk()  # frame i at (i, 0 or 0.5, 1)
        self.assertEqual(windows_in_area(kapture_data, sphere_area([5, 0, 1], 1.2), gap=0.15, pad=0),
                         [(frames(4)[0], frames(6)[0])])
        # the height counts: same center 3 m higher, out of reach
        self.assertEqual(windows_in_area(kapture_data, sphere_area([5, 0, 4], 1.2), gap=0.15, pad=0), [])
        self.assertEqual(windows_in_area(kapture_data, sphere_area([5, 0, 4], 3.2), gap=0.15, pad=0),
                         [(frames(4)[0], frames(6)[0])])

    def test_trajectory_positions(self):
        kapture_data = make_walk()
        timestamps, positions = trajectory_positions(kapture_data.trajectories)
        self.assertEqual(timestamps.tolist(), frames(*range(NB_FRAMES)))
        np.testing.assert_allclose(positions[3], [3, 0.5, 1], atol=1e-9)
        np.testing.assert_allclose(positions[:, 0], np.arange(NB_FRAMES), atol=1e-9)
        timestamps, positions = trajectory_positions(kapture_data.trajectories, ['rig'])
        self.assertEqual((len(timestamps), positions.shape), (0, (0, 3)))

    def test_windows_in_area(self):
        kapture_data = make_walk()
        box = box_area([2.5, 6.5, -1, 1, 0, 2])  # frames 3 .. 6, the rig is at z = 1
        self.assertEqual(windows_in_area(kapture_data, box, gap=0.15, pad=0), [(frames(3)[0], frames(6)[0])])
        self.assertEqual(windows_in_area(kapture_data, box, gap=0.15, pad=0.1), [(frames(2)[0], frames(7)[0])])
        # the odd frames only (y = 0.5): 0.2 s apart, one period when the gap allows it
        odd = box_area([2.5, 8.5, 0.4, 1, 0, 2])
        self.assertEqual(windows_in_area(kapture_data, odd, gap=0.15, pad=0),
                         [(t, t) for t in frames(3, 5, 7)])
        self.assertEqual(windows_in_area(kapture_data, odd, gap=0.25, pad=0), [(frames(3)[0], frames(7)[0])])
        # every axis counts: same box above the rig, or beside
        self.assertEqual(windows_in_area(kapture_data, box_area([2.5, 6.5, -1, 1, 2, 3]), gap=0.15, pad=0), [])
        self.assertEqual(windows_in_area(kapture_data, box_area([2.5, 6.5, None, None, None, None]), gap=0.15, pad=0),
                         [(frames(3)[0], frames(6)[0])])
        self.assertEqual(windows_in_area(kapture_data, box_area([None, 6.5, None, None, None, None]), gap=0.15, pad=0),
                         [(frames(0)[0], frames(6)[0])])
        self.assertEqual(windows_in_area(kapture_data, box_area([100, 101, 100, 101, 0, 2])), [])
        self.assertEqual(windows_in_area(kapture_data, box, skip_sensor_ids=['rig']), [])
        with self.assertRaises(ValueError):
            windows_in_area(kapture.Kapture(), box)


class TestKaptureExtractArea(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.input_path = path.join(self._tempdir.name, 'full')
        self.output_path = path.join(self._tempdir.name, 'area')
        write_kapture(self.input_path, make_walk())
        self.box = [2.5, 6.5, -1, 1, 0, 2]

    def tearDown(self):
        self._tempdir.cleanup()

    def test_extract_area(self):
        kapture_extract_area(self.input_path, self.output_path, self.box, gap=0.15, pad=0.05,
                             skip_sensor_ids=['lidar0_clean'], transfer=TransferAction.link_relative)
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(sorted(reloaded.records_camera), frames(3, 4, 5, 6))
        self.assertEqual(sorted(reloaded.trajectories), frames(3, 4, 5, 6))
        self.assertEqual(reloaded.records_lidar.sensors_ids, {'lidar0'})
        self.assertEqual(len(reloaded.records_gyroscope), 2 + 3 * 4 + 1 + 2)  # 0.05 s before and after
        self.assertIsNone(reloaded.points3d)
        link = get_record_fullpath(self.output_path, f'cam0/{frames(3)[0]}.jpg')
        self.assertTrue(path.islink(link) and path.isfile(link))
        with open(path.join(self.output_path, 'subset.json')) as file:
            subset = json.load(file)
        self.assertEqual(subset['box'], self.box)
        self.assertNotIn('sphere', subset)
        self.assertEqual(subset['windows'], [[frames(3)[0] - 50_000, frames(6)[0] + 50_000]])
        self.assertEqual((subset['gap'], subset['pad']), (0.15, 0.05))
        self.assertEqual(subset['records']['records_camera'], 8)

    def test_crop_points3d(self):
        kapture_extract_area(self.input_path, self.output_path, self.box, gap=0.15, pad=0, crop_points3d=True)
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(reloaded.points3d.shape, (4, 6))  # x = 3.25 .. 6.25, z = 0 (not z = 3)
        self.assertEqual(set(reloaded.points3d[:, 2]), {0})
        self.assertEqual(sorted(set(reloaded.points3d[:, 0])), [3.25, 4.25, 5.25, 6.25])
        self.assertEqual(sorted(set(reloaded.points3d[:, 5])), [3, 4, 5, 6])

    def test_box_without_bounds(self):
        kapture_extract_area(self.input_path, self.output_path, [2.5, 6.5, -1, 1, None, None], gap=0.15, pad=0,
                             crop_points3d=True)
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(sorted(reloaded.trajectories), frames(3, 4, 5, 6))
        self.assertEqual(reloaded.points3d.shape, (8, 6))  # both heights
        self.assertEqual(set(reloaded.points3d[:, 2]), {0, 3})
        with open(path.join(self.output_path, 'subset.json')) as file:
            self.assertEqual(json.load(file)['box'], [2.5, 6.5, -1, 1, None, None])

    def test_extract_sphere(self):
        kapture_extract_area(self.input_path, self.output_path, sphere=[5, 0, 1, 1.3], gap=0.15, pad=0,
                             crop_points3d=True)
        reloaded = kapture_from_dir(self.output_path)
        self.assertEqual(sorted(reloaded.records_camera), frames(4, 5, 6))
        self.assertEqual(sorted(reloaded.trajectories), frames(4, 5, 6))
        # points at (x + 0.25, 0.2, 0 or 3): within 1.3 m of (5, 0, 1) at z = 0 for x = 4, 5 only
        self.assertEqual(sorted(reloaded.points3d[:, 0]), [4.25, 5.25])
        self.assertEqual(set(reloaded.points3d[:, 2]), {0})
        with open(path.join(self.output_path, 'subset.json')) as file:
            subset = json.load(file)
        self.assertEqual(subset['sphere'], [5, 0, 1, 1.3])
        self.assertNotIn('box', subset)

    def test_one_area_only(self):
        with self.assertRaises(ValueError):
            kapture_extract_area(self.input_path, self.output_path)
        with self.assertRaises(ValueError):
            kapture_extract_area(self.input_path, self.output_path, self.box, sphere=[5, 0, 1, 1.2])
        with self.assertRaises(ValueError):
            kapture_extract_area(self.input_path, self.output_path, sphere=[5, 0, 1])
        with self.assertRaises(ValueError):
            kapture_extract_area(self.input_path, self.output_path, [2.5, 6.5, -1, 1])
        self.assertFalse(path.exists(self.output_path))

    def test_nothing_in_the_area(self):
        with self.assertRaises(ValueError):
            kapture_extract_area(self.input_path, self.output_path, [100, 101, 100, 101, 0, 2])
        with self.assertRaises(ValueError):
            kapture_extract_area(self.input_path, self.output_path, sphere=[100, 100, 0, 1])
        self.assertFalse(path.exists(self.output_path))


if __name__ == '__main__':
    unittest.main()
