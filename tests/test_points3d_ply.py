#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the points3d stored as PLY (reconstruction/points3d.ply), with optional per point timestamps,
and of the conversion between points3d.txt and points3d.ply.
"""

import os
import os.path as path
import tempfile
import unittest

import numpy as np

import kapture
import kapture.io.csv as csv
from kapture.algo.compare import equal_points3d
from kapture.algo.convert_points3d import kapture_convert_points3d
from kapture.algo.merge_reconstruction import merge_points3d
from kapture.io.ply import points3d_from_ply_file, points3d_to_ply_file, ply_vertex_count, read_ply, read_ply_header
from kapture.io.structure import delete_existing_kapture_files

EPOCH_NS = 1_762_950_000_123_456_789  # > 2^53: not exactly representable as a double


def make_points3d(nb: int = 5, colors: bool = True, timestamps: bool = False) -> kapture.Points3d:
    rng = np.random.default_rng(0)
    data = rng.uniform(-100, 100, (nb, 3))
    if colors:
        data = np.hstack([data, rng.integers(0, 256, (nb, 3)).astype(np.float64)])
    return kapture.Points3d(data, timestamps=EPOCH_NS + np.arange(nb) * 1_000_001 if timestamps else None)


def make_kapture(dirpath: str, points3d: kapture.Points3d, points3d_format: str = 'txt') -> None:
    sensors = kapture.Sensors()
    sensors['cam0'] = kapture.Camera(kapture.CameraType.SIMPLE_PINHOLE, [640, 480, 500, 320, 240])
    csv.kapture_to_dir(dirpath, kapture.Kapture(sensors=sensors, points3d=points3d), points3d_format=points3d_format)


class TestPoints3dTimestamps(unittest.TestCase):
    def test_attribute(self):
        points3d = make_points3d(timestamps=True)
        self.assertTrue(points3d.has_timestamps())
        self.assertEqual(points3d.timestamps.dtype, np.int64)
        self.assertFalse(make_points3d().has_timestamps())
        self.assertIsNone(points3d[1:].timestamps)  # not carried over by numpy operations
        np.testing.assert_array_equal(kapture.Points3d(points3d).timestamps, points3d.timestamps)  # but by copy
        with self.assertRaises(ValueError):
            kapture.Points3d(np.zeros((3, 3)), timestamps=[1, 2])
        with self.assertRaises(TypeError):
            kapture.Points3d(np.zeros((2, 3)), timestamps=[1., 2.])

    def test_merge_and_compare(self):
        points3d = make_points3d(timestamps=True)
        merged = merge_points3d([points3d, None, points3d])
        self.assertEqual(merged.shape, (10, 6))
        np.testing.assert_array_equal(merged.timestamps, np.concatenate([points3d.timestamps] * 2))
        self.assertIsNone(merge_points3d([points3d, make_points3d()]).timestamps)
        self.assertTrue(equal_points3d(points3d, kapture.Points3d(points3d)))
        self.assertFalse(equal_points3d(points3d, make_points3d()))
        other = kapture.Points3d(points3d, timestamps=points3d.timestamps + 1)
        self.assertFalse(equal_points3d(points3d, other))


class TestPoints3dPly(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.filepath = path.join(self._tempdir.name, 'reconstruction', 'points3d.ply')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_round_trip(self):
        for colors in (True, False):
            points3d = make_points3d(colors=colors)
            csv.points3d_to_file(self.filepath, points3d)  # dispatched on the extension
            reloaded = csv.points3d_from_file(self.filepath)
            self.assertEqual(reloaded.shape, points3d.shape)
            np.testing.assert_array_equal(reloaded.as_array(), points3d.as_array())  # float64 by default: no loss
            self.assertIsNone(reloaded.timestamps)
            self.assertFalse(path.exists(self.filepath + '.tmp'))
        self.assertEqual(ply_vertex_count(self.filepath), 5)

    def test_float32(self):
        points3d = make_points3d()
        points3d_to_ply_file(self.filepath, points3d, xyz_dtype=np.float32)
        self.assertEqual(read_ply(self.filepath).dtype['x'], np.float32)
        np.testing.assert_allclose(points3d_from_ply_file(self.filepath).as_array(), points3d.as_array(), atol=1e-4)

    def test_empty(self):
        points3d_to_ply_file(self.filepath, kapture.Points3d())
        self.assertEqual(points3d_from_ply_file(self.filepath).shape, (0, 6))

    def test_comments_and_atomic_write(self):
        from kapture.io.ply import write_ply
        from kapture.utils.paths import atomic_write
        points3d = kapture.Points3d(np.arange(12, dtype=float).reshape(2, 6))
        points3d_to_ply_file(self.filepath, points3d, comments=['made by a test', 'voxel_size=0.05'])
        comments = read_ply_header(self.filepath)['comments']
        self.assertIn('made by a test', comments)
        self.assertIn('voxel_size=0.05', comments)
        np.testing.assert_array_equal(points3d_from_ply_file(self.filepath).as_array(), points3d.as_array())
        self.assertFalse(os.path.exists(self.filepath + '.tmp'))  # written next to the file, then renamed
        with self.assertRaises(ValueError):
            write_ply(self.filepath, np.zeros(1, dtype=[('x', 'f4')]), comments=['two\nlines'])
        # an interrupted write keeps the previous file and leaves no temporary file
        before = open(self.filepath, 'rb').read()
        with self.assertRaises(RuntimeError):
            with atomic_write(self.filepath) as tmp_filepath:
                with open(tmp_filepath, 'wb') as f:
                    f.write(b'half')
                raise RuntimeError('interrupted')
        self.assertEqual(open(self.filepath, 'rb').read(), before)
        self.assertFalse(os.path.exists(self.filepath + '.tmp'))

    def test_extra_properties_ignored(self):
        from kapture.io.ply import write_ply
        vertices = np.zeros(3, dtype=[('intensity', 'f4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')])
        vertices['x'] = [1, 2, 3]
        write_ply(self.filepath, vertices)
        points3d = points3d_from_ply_file(self.filepath)
        self.assertEqual(points3d.shape, (3, 3))  # red alone is not a color
        np.testing.assert_array_equal(points3d.as_array()[:, 0], [1, 2, 3])
        with self.assertRaises(ValueError):
            write_ply(self.filepath, np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4')]))
            points3d_from_ply_file(self.filepath)

    def test_timestamps_relative(self):
        points3d = make_points3d(timestamps=True)
        points3d_to_ply_file(self.filepath, points3d)  # relative by default
        self.assertIn(f'timestamp_offset {EPOCH_NS}', read_ply_header(self.filepath)['comments'])
        self.assertEqual(read_ply(self.filepath).dtype['timestamp'], np.float64)
        reloaded = points3d_from_ply_file(self.filepath)
        np.testing.assert_array_equal(reloaded.timestamps, points3d.timestamps)  # exact

    def test_timestamps_absolute(self):
        points3d = make_points3d(timestamps=True)
        with self.assertLogs('ply', level='WARNING'):
            points3d_to_ply_file(self.filepath, points3d, timestamp_encoding='absolute')
        self.assertFalse(any(comment.startswith('timestamp_offset')
                             for comment in read_ply_header(self.filepath)['comments']))
        reloaded = points3d_from_ply_file(self.filepath)
        errors = np.abs(reloaded.timestamps - points3d.timestamps)
        self.assertTrue(errors.any())  # truncated
        self.assertLessEqual(errors.max(), 256)  # ulp of a double around 1.7e18 is 256
        with self.assertRaises(ValueError):
            points3d_to_ply_file(self.filepath, points3d, timestamp_encoding='whatever')

    def test_timestamps_lost_in_txt(self):
        filepath = path.join(self._tempdir.name, 'points3d.txt')
        with self.assertLogs(kapture.logger, level='WARNING'):
            csv.points3d_to_file(filepath, make_points3d(timestamps=True))
        self.assertIsNone(csv.points3d_from_file(filepath).timestamps)


class TestKapturePoints3dFormats(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.dirpath = self._tempdir.name
        self.txt = csv.get_points3d_fullpath(self.dirpath, 'txt')
        self.ply = csv.get_points3d_fullpath(self.dirpath, 'ply')

    def tearDown(self):
        self._tempdir.cleanup()

    def test_fullpath(self):
        self.assertIsNone(csv.get_points3d_fullpath(self.dirpath))
        self.assertIsNone(csv.get_points3d_format(self.dirpath))
        self.assertTrue(self.txt.endswith(path.join('reconstruction', 'points3d.txt')))
        with self.assertRaises(ValueError):
            csv.get_points3d_fullpath(self.dirpath, 'pcd')

    def test_to_and_from_dir(self):
        points3d = make_points3d(timestamps=True)
        for points3d_format in ('ply', 'txt'):
            make_kapture(self.dirpath, points3d, points3d_format)
            self.assertEqual(csv.get_points3d_format(self.dirpath), points3d_format)
            self.assertFalse(path.exists(self.ply if points3d_format == 'txt' else self.txt))  # no twin left
            reloaded = csv.kapture_from_dir(self.dirpath)
            np.testing.assert_allclose(reloaded.points3d.as_array(), points3d.as_array())
            self.assertEqual(reloaded.points3d.has_timestamps(), points3d_format == 'ply')
            self.assertEqual(csv.get_stored_points3d_number(self.dirpath), 5)
            self.assertIsNone(csv.kapture_from_dir(self.dirpath, skip_list=[kapture.Points3d]).points3d)

    def test_default_format_is_the_existing_one(self):
        make_kapture(self.dirpath, make_points3d(), 'ply')
        reloaded = csv.kapture_from_dir(self.dirpath)
        csv.kapture_to_dir(self.dirpath, reloaded)
        self.assertTrue(path.isfile(self.ply) and not path.exists(self.txt))
        other = path.join(self.dirpath, 'other')
        csv.kapture_to_dir(other, reloaded)  # nothing there: txt
        self.assertEqual(csv.get_points3d_format(other), 'txt')

    def test_both_files(self):
        make_kapture(self.dirpath, make_points3d(nb=3), 'txt')
        csv.points3d_to_file(self.ply, make_points3d(nb=4))
        with self.assertLogs(kapture.logger, level='WARNING'):
            self.assertEqual(len(csv.kapture_from_dir(self.dirpath).points3d), 4)  # ply first

    def test_delete_existing_kapture_files(self):
        make_kapture(self.dirpath, make_points3d(), 'ply')
        delete_existing_kapture_files(self.dirpath, force_erase=True)
        self.assertFalse(path.exists(self.ply))

    def test_convert(self):
        points3d = make_points3d()
        make_kapture(self.dirpath, points3d, 'txt')
        with open(self.txt) as file:
            original_txt = file.read()
        points3d = csv.points3d_from_file(self.txt)  # as written in text (%.10f)
        self.assertEqual(kapture_convert_points3d(self.dirpath, 'ply'), self.ply)
        self.assertFalse(path.exists(self.txt))
        np.testing.assert_array_equal(csv.points3d_from_file(self.ply).as_array(), points3d.as_array())  # same order
        kapture_convert_points3d(self.dirpath, 'txt', keep_source=True)
        self.assertTrue(path.isfile(self.ply))
        with open(self.txt) as file:
            self.assertEqual(file.read(), original_txt)
        with self.assertRaises(FileExistsError):
            kapture_convert_points3d(self.dirpath, 'ply')  # from txt, ply exists
        kapture_convert_points3d(self.dirpath, 'ply', xyz_dtype=np.float32, force=True)
        self.assertEqual(read_ply(self.ply).dtype['x'], np.float32)
        self.assertFalse(path.exists(self.txt))
        # ply to ply: re-encode
        kapture_convert_points3d(self.dirpath, 'ply')
        self.assertEqual(read_ply(self.ply).dtype['x'], np.float64)
        with self.assertRaises(ValueError):
            kapture_convert_points3d(self.dirpath, 'pcd')

    def test_convert_nothing(self):
        make_kapture(self.dirpath, None)
        self.assertIsNone(kapture_convert_points3d(self.dirpath, 'ply'))


if __name__ == '__main__':
    unittest.main()
