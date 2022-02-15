#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os
import os.path as path
import sys
import tempfile
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
import kapture.io.records
from kapture.io.binary import transfer_files_from_dir_link, transfer_files_from_dir_copy
from kapture.utils.paths import path_secure, populate_files_in_dirpath


def make_fake_filenames(root_path: str, post_fix=''):
    filenames = [
        path_secure(path.join(dir1, dir2, filename))
        for dir1 in ['a', 'b']
        for dir2 in [f'{i:02d}' for i in range(3)]
        for filename in [f'{i:02d}' for i in range(3)]
    ]
    filepaths = [path_secure(path.join(root_path, filename + post_fix)) for filename in filenames]
    for filepath in filepaths:
        os.makedirs(path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(filepath)
    return filenames


class TestRecordCopy(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._source_dirpath = path.join(self._tempdir.name, 'source')
        self._dest_dirpath = path.join(self._tempdir.name, 'dest')
        self._filenames = make_fake_filenames(self._source_dirpath)

    def tearDown(self):
        self._tempdir.cleanup()

    def test_populate(self):
        filepaths_retrieved = populate_files_in_dirpath(self._source_dirpath)
        self.assertEqual(set(self._filenames),
                         set(filepaths_retrieved))

    def test_copy_generators(self):
        origin_filepaths = (
            path_secure(path.join(self._source_dirpath, filename))
            for filename in self._filenames)
        expected_filepaths = (
            kapture.io.records.get_image_fullpath(self._dest_dirpath, filename)
            for filename in self._filenames)

        transfer_files_from_dir_copy(
            origin_filepaths,
            expected_filepaths
        )

        for expected_filepath in expected_filepaths:
            self.assertTrue(path.isfile(expected_filepath))
        for origin_filepath in origin_filepaths:
            self.assertTrue(path.isfile(origin_filepath))


class TestRecordLinkAbs(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._source_dirpath = path.join(self._tempdir.name, 'source')
        self._dest_dirpath = path.join(self._tempdir.name, 'dest')
        self._filenames = make_fake_filenames(self._source_dirpath)

    def tearDown(self):
        self._tempdir.cleanup()

    @unittest.skipIf(sys.platform.startswith("win"), "Do not work on Windows without admin rights")
    def test_link_abs(self):
        source_filepaths = [
            path_secure(path.join(self._source_dirpath, filename))
            for filename in self._filenames]
        destination_filepaths = [
            kapture.io.records.get_image_fullpath(self._dest_dirpath, filename)
            for filename in self._filenames]
        transfer_files_from_dir_link(
            source_filepaths, destination_filepaths, do_relative_link=False
        )

        for destination_filepath, source_filepath in zip(destination_filepaths, source_filepaths):
            self.assertTrue(path.islink(destination_filepath))
            resolved_path = os.readlink(destination_filepath)
            self.assertEqual(source_filepath, resolved_path)

        for source_filepath in source_filepaths:
            self.assertTrue(path.isfile(source_filepath))


class TestRecordLinkRel(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._source_dirpath = path.join(self._tempdir.name, 'source')
        self._dest_dirpath = path.join(self._tempdir.name, 'dest')
        self._filenames = make_fake_filenames(self._source_dirpath)

    def tearDown(self):
        self._tempdir.cleanup()

    @unittest.skipIf(sys.platform.startswith("win"), "Do not work on Windows without admin rights")
    def test_link_rel(self):
        source_filepaths = [
            path_secure(path.join(self._source_dirpath, filename))
            for filename in self._filenames]
        destination_filepaths = [
            kapture.io.records.get_image_fullpath(self._dest_dirpath, filename)
            for filename in self._filenames]
        transfer_files_from_dir_link(
            source_filepaths, destination_filepaths, do_relative_link=True
        )

        for destination_filepath, source_filepath in zip(destination_filepaths, source_filepaths):
            self.assertTrue(path.islink(destination_filepath))
            self.assertNotEqual(source_filepath, os.readlink(destination_filepath))
            resolved_path = path.normpath(
                path.join(path.dirname(destination_filepath), os.readlink(destination_filepath)))
            self.assertEqual(source_filepath, resolved_path)

        for source_filepath in source_filepaths:
            self.assertTrue(path.isfile(source_filepath))


class TestPathComputation(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self._kapture_path = self._tempdir.name

    def tearDown(self):
        self._tempdir.cleanup()

    def test_get_image_fullpath_empty(self):
        images_path = kapture.io.records.get_image_fullpath("")
        self.assertEqual(kapture.io.records.RECORD_DATA_DIRNAME, images_path)

    def test_get_image_fullpath(self):
        image_name = "my_image.jpg"
        image_path = kapture.io.records.get_image_fullpath(self._kapture_path, image_name)
        self.assertTrue(image_path.startswith(path_secure(self._kapture_path)), "Image path is under the kapture path")
        self.assertTrue(image_path.endswith(image_name), "Image path end with the image name")


if __name__ == '__main__':
    unittest.main()
