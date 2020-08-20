# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Inject GPS to EXIF
"""
from typing import Dict, Optional
import logging
from PIL import Image
import piexif
from fractions import Fraction
import kapture
from kapture.io.records import get_image_fullpath

logger = logging.getLogger('exif')


def write_exif(image_filepath: str,
               exif_data):
    """
    read_data reverse operation: convert back string to ID, and write it to file.

    """
    assert exif_data is not None
    image = Image.open(image_filepath)
    exif_bytes = piexif.dump(exif_data)
    image.save(image_filepath, exif=exif_bytes)


def update_exif(image_filepath: str,
                exif_data):
    exif_bytes = piexif.dump(exif_data)
    piexif.insert(exif_bytes, image_filepath)


def clear_exif(image_filepath: str):
    piexif.remove(image_filepath)


def convert_dms_from_float(value, loc):
    """
    convert floats coordinates into degrees, munutes and seconds + ref tuple

    return: tuple like (25, 13, 48.343 ,'N')
    """
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg = int(abs_value)
    t1 = (abs_value-deg) * 60
    min = int(t1)
    sec = round((t1 - min) * 60, 5)
    return deg, min, sec, loc_value


def convert_rational_from_float(number):
    """
    converts a float to rational as form of a tuple.
    """
    f = Fraction(str(number))  # str act as a round
    return f.numerator, f.denominator


def gps_record_to_exif_dict(
        gps_record: kapture.RecordGnss
) -> dict:
    """
    gps_record = x, y, z, utc, dop
    """

    lat_deg = convert_dms_from_float(gps_record.y, ["S", "N"])
    lng_deg = convert_dms_from_float(gps_record.x, ["W", "E"])
    latitude_hms = (convert_rational_from_float(lat_deg[0]), convert_rational_from_float(lat_deg[1]),
                    convert_rational_from_float(lat_deg[2]))
    longitude_hms = (convert_rational_from_float(lng_deg[0]), convert_rational_from_float(lng_deg[1]),
                     convert_rational_from_float(lng_deg[2]))
    gps_infos = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLongitude: longitude_hms,
            piexif.GPSIFD.GPSLongitudeRef: b'N',
            piexif.GPSIFD.GPSLatitude: latitude_hms,
            piexif.GPSIFD.GPSLatitudeRef: b'E',
            piexif.GPSIFD.GPSAltitude: convert_rational_from_float(round(gps_record.z, 1)),
            piexif.GPSIFD.GPSAltitudeRef: 1,
            piexif.GPSIFD.GPSTimeStamp: convert_rational_from_float(gps_record.utc),
            piexif.GPSIFD.GPSDOP: convert_rational_from_float(gps_record.dop)
    }
    exif_data = {'GPS': gps_infos}
    return exif_data


def export_gps_to_exif(
        kapture_data: kapture.Kapture,
        kapture_dirpath: str,
        gps_id_to_cam_id: Optional[Dict[str, str]] = None
):
    """
    Export GPS from GNSS data in kapture, to image exifs.
    If the mapping from gnns_id to camera_id is not given,
    gnss_id is assumed to be to be gps_<cam_id>.
    """

    # sanity check
    if None in [kapture_data.sensors, kapture_data.records_camera, kapture_data.records_gnss]:
        logger.warning('cannot export exif: missing data.')
        return

    # auto build GPS/camera map, based on the prefix rule.
    if gps_id_to_cam_id is None:
        camera_ids = {cam_id
                      for cam_id, sensor in kapture_data.sensors.items()
                      if sensor.sensor_type == 'camera'}

        gps_ids = {gps_id
                   for gps_id, sensor in kapture_data.sensors.items()
                   if sensor.sensor_type == 'gnss' and gps_id.startswith('GPS_')}

        gps_id_to_cam_id = {'GPS_' + cam_id: cam_id
                            for cam_id in camera_ids
                            if 'GPS_' + cam_id in gps_ids}

        if len(gps_id_to_cam_id) != len(gps_ids):
            logger.warning('unable to map some GPS to a camera.')

    gps_records = ((timestamp, gps_id, gps_id_to_cam_id[gps_id], gnss_record)
                   for timestamp, gps_id, gnss_record in kapture.flatten(kapture_data.records_gnss)
                   if gps_id in gps_id_to_cam_id)

    for timestamp, gps_id, cam_id, gps_record in gps_records:
        if (timestamp, cam_id) not in kapture_data.records_camera:
            logger.warning(f'no image found corresponding to GPS record ({timestamp}, {cam_id})')
        else:
            image_name = kapture_data.records_camera[timestamp, cam_id]
            image_filepath = get_image_fullpath(kapture_dir_path=kapture_dirpath, image_filename=image_name)
            exif_data = gps_record_to_exif_dict(gps_record)
            update_exif(image_filepath, exif_data)
