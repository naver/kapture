# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Inject GPS to EXIF
"""
from typing import Dict, Optional
import logging
from PIL import Image
import piexif

import kapture
from kapture.io.records import get_image_fullpath
from .import_exif import read_exif, replace_exif_id_by_names

logger = logging.getLogger('exif')


def write_exif(image_filepath: str,
               exif_data):
    """
    read_data reverse operation: convert back string to ID, and write it to file.

    """
    image = Image.open(image_filepath)
    exif_bytes = piexif.dump(exif_data)
    image.save(image_filepath, exif=exif_bytes)


def update_exif(image_filepath: str,
                exif_data):
    exif_data_merged = read_exif(image_filepath).update(exif_data)
    write_exif(image_filepath, exif_data_merged)


def clear_exif(image_filepath: str):
    piexif.remove(image_filepath)


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

    gps_records = ((timestamp, gps_id_to_cam_id[gps_id], gnss_record)
                   for timestamp, gps_id, gnss_record in kapture.flatten(kapture_data.records_gnss)
                   if gps_id in gps_id_to_cam_id)

    for timestamp, cam_id, gps_record in gps_records:
        if (timestamp, cam_id) not in kapture_data.records_camera:
            logger.warning(f'no image found corresponding to GPS record ({timestamp}, {cam_id})')
        else:
            image_name = kapture_data.records_camera[timestamp, cam_id]
            image_filepath = get_image_fullpath(kapture_dir_path=kapture_dirpath, image_filename=image_name)
            exif_data = read_exif(image_filepath)
            write_exif(image_filepath, exif_data)
