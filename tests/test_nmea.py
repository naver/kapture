#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os.path as path
import tempfile
import unittest
import numpy as np
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
from kapture.converter.nmea.import_nmea import extract_gnss_from_nmea


class TestImportNmea(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        test_dir_path = path.dirname(__file__)
        cls._samples_folder = path.abspath(path.join(test_dir_path, '../samples/4seasons'))
        cls._tmu_sample_path = path.abspath(path.join(cls._samples_folder, 'TUM'))
        cls._kapture_sample_path = path.abspath(path.join(cls._samples_folder, 'kapture'))
        cls._tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """
        Clean up after every test
        """
        self._tempdir.cleanup()

    def test_read_nmea(self):
        nmea_file_path = path.join(self._tmu_sample_path, 'recording_2020-10-07_14-47-51', 'septentrio.nmea')
        gnss_id = 'my_gps'
        kapture_sensors, records_gnss = extract_gnss_from_nmea(nmea_file_path, gnss_id)

        # check sensor part
        self.assertIsInstance(kapture_sensors, kapture.Sensors)
        self.assertEqual(len(kapture_sensors), 1)
        self.assertIn(gnss_id, kapture_sensors)
        gnss_sensor = kapture_sensors[gnss_id]
        self.assertEqual(gnss_sensor.sensor_type, kapture.SensorType.gnss.name)
        self.assertEqual(gnss_sensor.sensor_params[0], 'EPSG:4326')
        self.assertEqual(gnss_sensor.name, gnss_id)

        # check records
        self.assertIsInstance(records_gnss, kapture.RecordsGnss)
        self.assertEqual(len(records_gnss), 3413)
        timestamps = np.array([int(k) for k in records_gnss.keys()])
        timestamp_range = np.min(timestamps), np.max(timestamps)
        self.assertEqual(timestamp_range, (1602074797700000000, 1602075208800000000))
        first_records = records_gnss[1602074797700000000]
        self.assertIn(gnss_id, first_records)
        first_record = first_records[gnss_id]
        first_record_expected = kapture.RecordGnss(x=11.619441895, y=48.19602170666667, z=494.6412,
                                                   utc=1602074797700000000, dop=2.6)
        self.assertEqual(first_record, first_record_expected)
