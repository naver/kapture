#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import shutil
import unittest
import piexif

# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.io.records import images_to_filepaths, get_image_fullpath
from kapture.converter.exif.import_exif import convert_gps_to_kapture_record, extract_gps_from_exif, \
    import_gps_from_exif, read_exif
from kapture.converter.exif.export_exif import export_gps_to_exif, write_exif, clear_exif

from kapture.algo.compare import equal_kapture, equal_records_gnss


class TestImportExif(unittest.TestCase):

    def setUp(self):
        self._samples_folder = path.abspath(path.join(path.dirname(__file__), '../samples/'))
        self._kapture_dirpath = path.join(self._samples_folder, 'berlin', 'kapture')
        self._kapture_data = kapture.io.csv.kapture_from_dir(self._kapture_dirpath)
        self._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tempdir.cleanup()

    def test_read_exif_invalid(self):
        invalid_image_filepath = path.join(self._samples_folder,
                                           '7scenes/microsoft/stairs/seq-01/frame-000000.color.jpg')
        exif_data = read_exif(invalid_image_filepath)
        self.assertIsNone(exif_data)

    def test_read_exif(self):
        image_filepath_w_exif = path.join(self._samples_folder, 'berlin/opensfm/images/01.jpg')
        exif_data = read_exif(image_filepath_w_exif)
        self.assertIsInstance(exif_data, dict)
        self.assertEqual(6, len(exif_data))
        self.assertIn('GPS', exif_data)

    def test_read_exif_gps(self):
        image_filepath_w_exif = path.join(self._samples_folder, 'berlin/opensfm/images/01.jpg')
        exif_data = read_exif(image_filepath_w_exif)
        self.assertIn('GPS', exif_data)
        gps_data = exif_data['GPS']
        self.assertIsInstance(gps_data, dict)
        self.assertEqual(6, len(gps_data))
        self.assertIn(piexif.GPSIFD.GPSLatitude, gps_data)

    def test_extract(self):
        sensors_gnss, records_gnss = extract_gps_from_exif(self._kapture_data, self._kapture_dirpath)
        # check sensors
        gps_id = 'GPS_v2 apple iphone 4s back camera 4.28mm f/2.4 3264 2448 perspective 0.9722'
        self.assertEqual(1, len(sensors_gnss))
        self.assertTrue(all(gnss_id.startswith('GPS_') for gnss_id in sensors_gnss.keys()))
        self.assertIn(gps_id, sensors_gnss)
        sensor_gnss = sensors_gnss[gps_id]
        self.assertEqual(kapture.SensorType.gnss.name, sensor_gnss.sensor_type)
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
        import_gps_from_exif(temp_kapture_dirpath)
        # reload
        temp_kapture_data = kapture.io.csv.kapture_from_dir(temp_kapture_dirpath)
        self.assertTrue(equal_kapture(temp_kapture_data, self._kapture_data))

    def test_write_exif(self):
        image_filepath_wo_exif = path.join(self._samples_folder,
                                           '7scenes/microsoft/stairs/seq-01/frame-000000.color.jpg')
        image_filepath_w_exif = path.join(self._samples_folder, 'berlin/opensfm/images/01.jpg')
        image_filepath_temp = path.join(self._tempdir.name, 'frame-000000.color.jpg')
        shutil.copy(image_filepath_wo_exif, image_filepath_temp)
        expected_exif = read_exif(image_filepath_w_exif)
        write_exif(image_filepath_temp, expected_exif)
        actual_exif = read_exif(image_filepath_temp)
        # self.assertEqual(len(expected_exif), len(actual_exif))
        self.assertEqual(set(expected_exif.keys()), set(actual_exif.keys()))
        self.assertDictEqual(expected_exif['GPS'], actual_exif['GPS'])

    def test_clear_exif(self):
        image_filepath_w_exif = path.join(self._samples_folder, 'berlin/opensfm/images/01.jpg')
        image_filepath_temp = path.join(self._tempdir.name, 'image.jpg')
        shutil.copy(image_filepath_w_exif, image_filepath_temp)
        clear_exif(image_filepath_temp)
        exif_data = read_exif(image_filepath_temp)
        self.assertIsNone(exif_data)

    def test_export(self):
        temp_kapture_dirpath = path.join(self._tempdir.name, 'kapture')
        shutil.copytree(self._kapture_dirpath, temp_kapture_dirpath)
        kapture_data = kapture.io.csv.kapture_from_dir(temp_kapture_dirpath)
        images_filepaths = images_to_filepaths(kapture_data.records_camera, temp_kapture_dirpath)
        # make sure there is no EXIF in images
        for image_filepath in images_filepaths.values():
            clear_exif(image_filepath)

        # insert gps to exif
        export_gps_to_exif(kapture_data, temp_kapture_dirpath)

        rebuilt_records = kapture.RecordsGnss()
        for timestamp, cam_id, image_name in kapture.flatten(kapture_data.records_camera):
            image_filepath = get_image_fullpath(temp_kapture_dirpath, image_name)
            exif_data = read_exif(image_filepath)
            rebuilt_records[timestamp, 'GPS_' + cam_id] = convert_gps_to_kapture_record(exif_data)

        self.assertTrue(equal_records_gnss(kapture_data.records_gnss, rebuilt_records))


if __name__ == '__main__':
    unittest.main()
