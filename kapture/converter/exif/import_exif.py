# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Extract GPS from EXIF
"""
from typing import Optional
from PIL import Image
import piexif
import logging
from typing import Tuple, Dict
from tqdm import tqdm
import kapture
from kapture.io.csv import get_csv_fullpath
from kapture.io.records import images_to_filepaths
from warnings import warn


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
        image_filepath: str
) -> Dict:
    """
    Reads the EXIF metatdata from image file using piexif.

    :param image_filepath:
    :return:
    """
    with Image.open(image_filepath) as image:
        if 'exif' not in image.info:
            return None
        exif_data = piexif.load(image.info['exif'])
    return exif_data


def convert_rational_to_float(
        rational_num: Tuple[float]
) -> float:
    assert isinstance(rational_num, tuple)
    return float(rational_num[0] / rational_num[1])


def convert_dms_to_float(
        dms: Tuple[Tuple[float]],
        reference: Optional[str] = None
) -> float:
    """
    Convert the degree, minute, seconds + ref representation to single float.

    :param dms: Is a triplet (tuple) of IFDRational.
    :param reference:
    :return:
    """
    # sanity check
    assert isinstance(dms, tuple)
    assert len(dms) == 3
    if any(not isinstance(v, tuple) for v in dms):
        # Pillow 3.1 uses IFDRational, but piexif tuple[float].
        warn('Make sure you have installed piexif==1.1.3')
        raise TypeError('Expect Tuple[Tuplep[float] for sexagesimal. Make sure you have installed piexif==1.1.3')

    assert all(isinstance(v, tuple) for v in dms)
    decimal = convert_rational_to_float(dms[0]) + \
        convert_rational_to_float(dms[1]) / 60 + \
        convert_rational_to_float(dms[2]) / 3600
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
        exif_data['GPS'] = {
            'piexif.GPSIFD.GPSLatitudeRef': 'N',
            'piexif.GPSIFD.GPSLatitude': ((52, 1), (31, 1), (810, 100)),
            'piexif.GPSIFD.GPSLongitudeRef': 'E',
            'piexif.GPSIFD.GPSLongitude': ((13, 1), (24, 1), (106, 100)),
            'piexif.GPSIFD.GPSAltitude': (27, 1),
            'piexif.GPSIFD.GPSDOP': (5, 1)
        }

    :return:
    """

    if 'GPS' not in exif_data:
        return kapture.RecordGnss()

    gps_info = exif_data['GPS']
    position = dict()
    position['x'] = convert_dms_to_float(
        dms=gps_info[piexif.GPSIFD.GPSLongitude],
        reference=gps_info.get(piexif.GPSIFD.GPSLongitudeRef)
    )
    position['y'] = convert_dms_to_float(
        dms=gps_info[piexif.GPSIFD.GPSLatitude],
        reference=gps_info.get(piexif.GPSIFD.GPSLatitudeRef)
    )
    position['z'] = convert_rational_to_float(
        rational_num=gps_info.get(piexif.GPSIFD.GPSAltitude, 0.0))
    position['dop'] = convert_rational_to_float(
        rational_num=gps_info.get(piexif.GPSIFD.GPSDOP, 0.0))
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
        gps_id = cam_to_gps_id[cam_id]
        exif_data = read_exif(image_filepath)
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
    logger.info('loading kapture partial ...')
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
