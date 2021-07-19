# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Extract GNSS data from NMEA 0183 files
"""
import re
from typing import Optional
from PIL import Image
from datetime import datetime, timezone, timedelta
import pytz
import logging
from typing import Tuple, Dict
from tqdm import tqdm
import kapture
from kapture.io.csv import table_from_file
from kapture.io.records import images_to_filepaths
from warnings import warn

logger = logging.getLogger('nmea')


def day_from_nmea_str(date_str: str) -> datetime:
    return datetime.strptime(date_str, '%d%m%y')


time_parser = re.compile(r'^(?P<hours>\d{2})(?P<minutes>\d{2})(?P<seconds>\d{2})\.(?P<milliseconds>\d+)$')


def nmea_str_to_datetime(
        day: datetime,
        time_str: str) -> datetime:
    """

    :param day: current date (assuming time part is set to 00:00:00)
    :param time_str: a string representing the nmea time (eg 124638.20 for 12h 46m 38s 200ms)
    :return: a datetime object with timestamp and timezone set.
    """
    time_params = time_parser.match(time_str)
    assert time_params is not None
    assert day.hour == 0
    time_params = time_params.groupdict()
    if len(time_params['milliseconds']) < 3:  # make sure milliseconds are on 3 digits
        time_params['milliseconds'] += '0' * (3 - len(time_params['milliseconds']))
    time_params = {k: int(v) for k, v in time_params.items()}
    time_of_day = timedelta(**time_params)
    timestamp = day + time_of_day
    timestamp = pytz.UTC.localize(timestamp)  # timestamp is guaranteed UTC
    return timestamp


def nmea_coord_to_lla(
    lat_str: str, lat_cardinal: str,
    lon_str: str, lon_cardinal: str,
    alt_str: str
) -> Tuple[float, float, float]:
    """
    converts the nmea quintuplet (4811.7605691,N,01137.1631304,E,494.6412) to decimal gps coordinates
    :param lat_str: latitude in nmea format
    :param lat_cardinal: latitude cardinal (N for north, S for south)
    :param lon_str: longitude in nmea format
    :param lon_cardinal: logitude cardinal (E for east, W for west)
    :param alt_str: altitude in nmea format
    :return:
    """
    lat_deg, lat_min = float(lat_str[0:2]), float(lat_str[2:])
    lat_dec = lat_deg + lat_min / 60.0
    lon_split = lon_str.index('.') - 2  # degree part may be 2 or 3 digits long
    lon_deg, lon_min = float(lon_str[0:lon_split]), float(lon_str[lon_split:])
    lon_dec = lon_deg + lon_min / 60.0
    alt_dec = float(alt_str)
    if 'S' == lat_cardinal:
        lat_dec *= -1
    if 'W' == lon_cardinal:
        lon_dec *= -1

    return lat_dec, lon_dec, alt_dec


def extract_gps_from_nmea(
        nmea_file_path: str
):
    """
    Extract GPS coordinates from NMEA file, returns the new sensor and gnss records.
    - Gnss timestamps are:
    - sensor ids are:

    :param nmea_file_path: path to nmea input file..
    :return:
    """
    # only load sensors + records_data:
    disable_tqdm = logger.getEffectiveLevel() != logging.INFO

    # make up new gps ids
    GPS_ID = 'gps01'

    # add new gps ids to sensors
    gnss_kapture_sensors = kapture.Sensors()
    gnss_kapture_sensors[GPS_ID] = kapture.Sensor(
        sensor_type='gnss', name=GPS_ID,
        sensor_params=['EPSG:4326'])  # aka WGS84
    #  x, y, z, utc, dop
    records_gnss = kapture.RecordsGnss()
    day = None

    with open(nmea_file_path, 'rt') as file:
        table = table_from_file(file)
        for trame in table:
            trame_type = trame[0]
            if '$GPRMC' == trame_type:  # retrieve date only (dont bother with coords, no altitude available)
                time_str, date_str = trame[1], trame[9]
                day = day_from_nmea_str(date_str)

            if day is None:  # dont bother do the rest if date is not retrieved yet.
                continue
            if '$GPGGA' == trame_type:  # GPS coordinates
                # UTC time, Latitude, hemisphere, Longitude, meridian, 1 (for GPS), #satellites used, HDOP, Altitude,...
                time_str, lat, ns, lon, ew, t, sats, dop, alt = trame[1:10]
                timestamp_dt = nmea_str_to_datetime(day, time_str)
                timestamp_ns = int(timestamp_dt.timestamp() * 1e9)
                # todo: apply time shift from gps to camera
                lat, lon, alt = nmea_coord_to_lla(lat, ns, lon, ew, alt)
                # x, y, z, utc, dop
                gps_record = kapture.RecordGnss(x=lon, y=lat, z=alt, utc=timestamp_ns, dop=dop)
                records_gnss[timestamp_ns, GPS_ID] = gps_record

    return gnss_kapture_sensors, records_gnss


def import_gps_from_nmea(
        nmea_file_path: str,
        kapture_dir_path: str
):
    """
    Imports (extracts and writes) GPS data from NMEA files

    :param nmea_file_path: path to nmea input file..
    :param kapture_dir_path: path to kapture directory. kapture data are modified.
    :return:
    """
    logger.info('loading kapture partial ...')

    # load exifs
    gnss_kapture_sensors, records_gnss = extract_gps_from_nmea(nmea_file_path)

    # kapture_data.sensors.update(gnss_kapture_sensors)

    # overwrite sensors and gnss only
    # sensors_filepath = get_csv_fullpath(kapture.Sensors, kapture_dirpath)
    # logger.info(f'writing {sensors_filepath} ...')
    # kapture.io.csv.sensors_to_file(sensors_filepath, kapture_data.sensors)
    # records_gnss_filepath = get_csv_fullpath(kapture.RecordsGnss, kapture_dirpath)
    # logger.info(f'writing {records_gnss_filepath} ...')
    # kapture.io.csv.records_gnss_to_file(records_gnss_filepath, records_gnss)
