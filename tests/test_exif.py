#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import shutil
import unittest
from PIL import Image

# kapture
import path_to_kapture  # enables import kapture
import kapture
import kapture.converter.exif.import_exif as exif
from kapture.algo.compare import equal_kapture


class TestImportExif(unittest.TestCase):

    def setUp(self):
        self._samples_folder = path.abspath(path.join(path.dirname(__file__), '../samples/'))
        self._kapture_dirpath = path.join(self._samples_folder, 'berlin', 'kapture')
        self._kapture_data = kapture.io.csv.kapture_from_dir(self._kapture_dirpath)
        self._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tempdir.cleanup()

    def test_read_exif_invalid(self):
        invalid_image_filepath = path.join(self._samples_folder, 'chess/7scenes/frame-000000.color.jpg')
        invalid_image = Image.open(invalid_image_filepath)
        exif_data = exif.read_exif(invalid_image)
        self.assertIsNone(exif_data)
        invalid_image.close()

    def test_read_exif(self):
        image_filepaths = kapture.io.records.images_to_filepaths(self._kapture_data.records_camera,
                                                                 self._kapture_dirpath)
        image = Image.open(image_filepaths['01.jpg'])
        exif_data = exif.read_exif(image)
        self.assertIsInstance(exif_data, dict)
        self.assertIn('ExifVersion', exif_data)
        self.assertEqual(35, len(exif_data))
        image.close()

    def test_read_exif_gps(self):
        image_filepaths = kapture.io.records.images_to_filepaths(self._kapture_data.records_camera,
                                                                 self._kapture_dirpath)
        image = Image.open(image_filepaths['01.jpg'])
        exif_data = exif.read_exif(image)
        self.assertIn('GPSInfo', exif_data)
        self.assertIsInstance(exif_data['GPSInfo'], dict)
        self.assertEqual(6, len(exif_data['GPSInfo']))
        self.assertIn('GPSLatitude', exif_data['GPSInfo'])
        image.close()

    def test_extract(self):
        sensors_gnss, records_gnss = exif.extract_gps_from_exif(kapture_data=self._kapture_data,
                                                                kapture_dirpath=self._kapture_dirpath)
        # check sensors
        gps_id = 'GPS_v2 apple iphone 4s back camera 4.28mm f/2.4 3264 2448 perspective 0.9722'
        self.assertEqual(1, len(sensors_gnss))
        self.assertTrue(all(gnss_id.startswith('GPS_') for gnss_id in sensors_gnss.keys()))
        self.assertIn(gps_id, sensors_gnss)
        sensor_gnss = sensors_gnss[gps_id]
        self.assertEqual('gnss', sensor_gnss.sensor_type)
        self.assertEqual(['EPSG:4326'], sensor_gnss.sensor_params)
        # check gps track
        self.assertEqual(3, len(records_gnss))
        first_pos = records_gnss[0, gps_id]
        expected_pos = kapture.RecordGnss(13.400294444444445, 52.51891666666666, 27.0, 0, 5.0)
        self.assertEqual(first_pos, expected_pos)

    def test_import(self):
        temp_kapture_dirpath = path.join(self._tempdir.name, 'kapture')
        shutil.copytree(self._kapture_dirpath, temp_kapture_dirpath)
        # remove all GPS data
        temp_kapture_data = kapture.io.csv.kapture_from_dir(temp_kapture_dirpath)
        temp_kapture_data.records_gnss = None
        for gps_id in [k for k in temp_kapture_data.sensors.keys() if k.startswith('GPS_')]:
            del temp_kapture_data.sensors[gps_id]
        kapture.io.csv.kapture_to_dir(temp_kapture_dirpath, temp_kapture_data)
        # do the import
        exif.import_gps_from_exif(temp_kapture_dirpath)
        # reload
        temp_kapture_data = kapture.io.csv.kapture_from_dir(temp_kapture_dirpath)
        self.assertTrue(equal_kapture(temp_kapture_data, self._kapture_data))


if __name__ == '__main__':
    unittest.main()
