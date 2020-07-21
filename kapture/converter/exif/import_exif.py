# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Extract GPS from EXIF
"""
from typing import Optional
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from PIL.ExifTags import TAGS, GPSTAGS
import logging
from typing import Tuple, Dict
from tqdm import tqdm
import kapture
from kapture.io.csv import get_csv_fullpath, records_camera_from_file
from kapture.io.records import images_to_filepaths
from warnings import warn


GPSTAGS_INV = {value: key for key, value in GPSTAGS.items()}

logger = logging.getLogger('exif')


def replace_exif_id_by_names(exif_dict: dict,
                             tag_dict: dict):
    """
    substitute TAG_ID with tag names when tag is referenced in tag_dict.

    :param exif_dict:
    :param tag_dict:
    :return:
    """
    return {
        tag_dict.get(key_id, key_id): value
        for key_id, value in exif_dict.items()
        if key_id in tag_dict
    }


def read_exif(
        image: Image
) -> Dict:
    """
    Reads the EXIF metatdata from PIL image and substitute known TAG_ID with strings.

    :param image:
    :return:
    """
    exif_data = image._getexif()
    if exif_data is not None:
        exif_data = replace_exif_id_by_names(exif_data, TAGS)
        if 'GPSInfo' in exif_data:
            exif_data['GPSInfo'] = replace_exif_id_by_names(exif_data['GPSInfo'], GPSTAGS)
    return exif_data


def convert_single_sexagesimal_to_decimal(
        sexagesimal: IFDRational
) -> float:
    assert isinstance(sexagesimal, IFDRational)
    return float(sexagesimal)


def convert_coordinates_sexagesimal_to_decimal(
        sexagesimal: Tuple[IFDRational],
        reference: Optional[str] = None
) -> float:
    """
    Convert the sexagesimal representation coordinate (degree, minute, seconds) to single float.

    :param sexagesimal: Is a triplet (tuple) of IFDRational.
    :param reference:
    :return:
    """
    # sanity check
    assert isinstance(sexagesimal, tuple)
    assert len(sexagesimal) == 3
    if any(not isinstance(v, IFDRational) for v in sexagesimal):
        # from Pillow 3.1, any single item tuples have been unwrapped and return a bare element.
        warn('Make sure you have installed Pillow>3.1')
        raise TypeError('Expect Type[IFDRational] for sexagesimal. Make sure you have installed Pillow>3.1')

    assert all(isinstance(v, IFDRational) for v in sexagesimal)
    decimal = convert_single_sexagesimal_to_decimal(sexagesimal[0]) + \
              convert_single_sexagesimal_to_decimal(sexagesimal[1]) / 60 + \
              convert_single_sexagesimal_to_decimal(sexagesimal[2]) / 3600
    # handle South or west coordinates.
    if reference is not None and reference.upper() in ['S', 'W']:
        decimal *= -1
    return decimal


def convert_gps_to_kapture_record(
        exif_data: dict
) -> kapture.RecordGnss:
    """
    Converts exif_data to kapture single RecordGnss.

    :param exif_data: input exif_data in EXIF format with tag nam (as given by read_exif).
        exif_data['GPSInfo'] = {
            'GPSLatitudeRef': 'N',
            'GPSLatitude': ((52, 1), (31, 1), (810, 100)),
            'GPSLongitudeRef': 'E',
            'GPSLongitude': ((13, 1), (24, 1), (106, 100)),
            'GPSAltitude': (27, 1),
            'GPSDOP': (5, 1)
        }

    :return:
    """

    if 'GPSInfo' not in exif_data:
        return kapture.RecordGnss()

    gps_info = exif_data['GPSInfo']
    position = {
        axis: convert_coordinates_sexagesimal_to_decimal(
            sexagesimal=gps_info['GPS' + axis_name], reference=gps_info.get('GPS' + axis_name + 'Ref'))
        for axis, axis_name in [('y', 'Latitude'), ('x', 'Longitude')]
    }
    position['z'] = convert_single_sexagesimal_to_decimal(gps_info['GPSAltitude']) if 'GPSAltitude' in gps_info else 0.
    position['dop'] = convert_single_sexagesimal_to_decimal(gps_info['GPSDOP']) if 'GPSDOP' in gps_info else 0.0
    position['utc'] = 0.

    return kapture.RecordGnss(**position)


def extract_gps_from_exif(
        kapture_data: kapture.Kapture,
        kapture_dirpath: str
):
    """
    Extract GPS coordinates from kapture dataset, returns the new sensor and gnss records.
    Gnss timestamps and sensor ids are guessed from timestamps and camera_id from images.
    The GNSS sensor_id are built prefixing 'GPS_'<cam_id>, with cam_id the sensor_id of the corresponding camera.

    :param kapture_data: input kapture data, must contains sensors and records_camera.
    :param kapture_dirpath: input path to kapture directory.
    :return:
    """
    # only load sensors + records_data:
    disable_tqdm = logger.getEffectiveLevel() != logging.INFO

    # make up new gps ids
    cam_to_gps_id = {  # cam_id -> gps_id
        cam_id: 'GPS_' + cam_id
        for cam_id, sensor in kapture_data.sensors.items()
        if sensor.sensor_type == 'camera'
    }  # cam_id -> gps_id

    # set all gps to EPSG:4326
    gps_epsg_codes = {gps_id: 'EPSG:4326' for gps_id in cam_to_gps_id.values()}
    # add new gps ids to sensors
    gnss_kapture_sensors = kapture.Sensors()
    for gps_id, epsg in gps_epsg_codes.items():
        gnss_kapture_sensors[gps_id] = kapture.Sensor(sensor_type='gnss', sensor_params=[epsg])

    image_filepaths = images_to_filepaths(kapture_data.records_camera, kapture_dirpath=kapture_dirpath)
    records_gnss = kapture.RecordsGnss()

    for timestamp, cam_id, image_name in tqdm(kapture.flatten(kapture_data.records_camera), disable=disable_tqdm):
        image_filepath = image_filepaths[image_name]
        logger.debug(f'extracting GPS tags from {image_filepath}')
        with Image.open(image_filepath) as image:
            gps_id = cam_to_gps_id[cam_id]
            exif_data = read_exif(image)
        gps_record = convert_gps_to_kapture_record(exif_data)
        records_gnss[timestamp, gps_id] = gps_record

    return gnss_kapture_sensors, records_gnss


def import_gps_from_exif(
        kapture_dirpath: str
):
    """
    Imports (extracts and writes) GPS data from EXIF metadata inside a kapture directory

    :param kapture_dirpath: path to kapture directory. kpautre data are modified.
    :return:
    """
    logger.info(f'loading kapture partial ...')
    skip_heavy_useless = {kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations}
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_dirpath, skip_list=skip_heavy_useless)

    # load exifs
    gnss_kapture_sensors, records_gnss = extract_gps_from_exif(kapture_data, kapture_dirpath)
    kapture_data.sensors.update(gnss_kapture_sensors)

    # overwrite sensors and gnss only
    sensors_filepath = get_csv_fullpath(kapture.Sensors, kapture_dirpath)
    logger.info(f'writing {sensors_filepath} ...')
    kapture.io.csv.sensors_to_file(sensors_filepath, kapture_data.sensors)
    records_gnss_filepath = get_csv_fullpath(kapture.RecordsGnss, kapture_dirpath)
    logger.info(f'writing {records_gnss_filepath} ...')
    kapture.io.csv.records_gnss_to_file(records_gnss_filepath, records_gnss)
