# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Extract GNSS data from NMEA 0183 files
"""
from datetime import datetime, timedelta
import logging
import pytz
import re
from typing import Tuple, Optional

import kapture
from kapture.io.csv import table_from_file

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
    :param lon_cardinal: longitude cardinal (E for east, W for west)
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


def extract_gnss_from_nmea(
        nmea_file_path: str,
        gnss_id: Optional[str] = 'gnss'
):
    """
    Extract coordinates from NMEA file, returns sensors with the new gnss sensor in it and gnss records.
    - Gnss timestamps are: in nanoseconds

    :param nmea_file_path: path to nmea input file..
    :param gnss_id: ID of the GNSS sensor.
    :return:
    """
    # only load sensors + records_data:

    # add new gps ids to sensors
    gnss_kapture_sensors = kapture.Sensors()
    gnss_kapture_sensors[gnss_id] = kapture.Sensor(
        sensor_type='gnss', name=gnss_id,
        sensor_params=['EPSG:4326'])  # aka WGS84
    #  x, y, z, utc, dop
    records_gnss = kapture.RecordsGnss()
    day = None

    with open(nmea_file_path, 'rt') as file:
        table = table_from_file(file)
        for trame in table:
            trame_type = trame[0]
            if '$GPRMC' == trame_type:  # retrieve date only (don't bother with coords, no altitude available)
                time_str, date_str = trame[1], trame[9]
                day = day_from_nmea_str(date_str)

            if day is None:  # don't bother do the rest if date is not retrieved yet.
                continue
            if '$GPGGA' == trame_type:  # GPS coordinates
                # UTC time, Latitude, hemisphere, Longitude, meridian, 1 (for GPS), #satellites used, HDOP, Altitude,...
                time_str, lat, ns, lon, ew, t, sats, dop, alt = trame[1:10]
                try:
                    timestamp_dt = nmea_str_to_datetime(day, time_str)
                    timestamp_ns = int(timestamp_dt.timestamp() * 1e9)
                    lat, lon, alt = nmea_coord_to_lla(lat, ns, lon, ew, alt)
                    # x, y, z, utc, dop
                    gps_record = kapture.RecordGnss(x=lon, y=lat, z=alt, utc=timestamp_ns, dop=dop)
                    records_gnss[timestamp_ns, gnss_id] = gps_record
                except Exception as e:
                    logger.debug(f'skipping malformed nmea frame {e}')

    return gnss_kapture_sensors, records_gnss
