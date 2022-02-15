# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
All reading and writing operations of kapture objects in CSV like files
"""

from collections import namedtuple
import datetime
import io
import numpy as np
import quaternion
import os
import os.path as path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Type, Union


import kapture
import kapture.io.features
from kapture.utils.logging import getLogger
from kapture.io.tar import KAPTURE_TARABLE_TYPES, TarCollection, TarHandler
from kapture.io.tar import get_feature_tar_fullpath, retrieve_tar_handler_from_collection

logger = kapture.logger

# file names conventions
CSV_FILENAMES = {
    kapture.Sensors: path.join('sensors', 'sensors.txt'),
    kapture.Trajectories: path.join('sensors', 'trajectories.txt'),
    kapture.Rigs: path.join('sensors', 'rigs.txt'),
    kapture.RecordsCamera: path.join('sensors', 'records_camera.txt'),
    kapture.RecordsDepth: path.join('sensors', 'records_depth.txt'),
    kapture.RecordsLidar: path.join('sensors', 'records_lidar.txt'),
    kapture.RecordsWifi: path.join('sensors', 'records_wifi.txt'),
    kapture.RecordsBluetooth: path.join('sensors', 'records_bluetooth.txt'),
    kapture.RecordsGnss: path.join('sensors', 'records_gnss.txt'),
    kapture.RecordsAccelerometer: path.join('sensors', 'records_accelerometer.txt'),
    kapture.RecordsGyroscope: path.join('sensors', 'records_gyroscope.txt'),
    kapture.RecordsMagnetic: path.join('sensors', 'records_magnetic.txt'),
    kapture.Points3d: path.join('reconstruction', 'points3d.txt'),
    kapture.Observations: path.join('reconstruction', 'observations.txt'),
}

FEATURES_CSV_FILENAMES = {
    kapture.Keypoints: lambda x: path.join('reconstruction', 'keypoints', x, 'keypoints.txt'),
    kapture.Descriptors: lambda x: path.join('reconstruction', 'descriptors', x, 'descriptors.txt'),
    kapture.GlobalFeatures: lambda x: path.join('reconstruction', 'global_features', x, 'global_features.txt'),
}


def get_csv_fullpath(kapture_type: Any, kapture_dirpath: str = '') -> str:
    """
    Returns the full path to csv kapture file for a given data structure and root directory.
    This path is the concatenation of the kapture root path and subpath into kapture into data structure.

    :param kapture_type: type of kapture data (kapture.RecordsCamera, kapture.Trajectories, ...)
    :param kapture_dirpath: root kapture path
    :return: full path of csv file for that type of data
    """
    assert kapture_type in CSV_FILENAMES
    filename = CSV_FILENAMES[kapture_type]
    return path.join(kapture_dirpath, filename)


def get_feature_csv_fullpath(kapture_type: Any, feature_name: str, kapture_dirpath: str = '') -> str:
    """
    Returns the full path to csv kapture file for a given datastructure and root directory.
    This path is the concatenation of the kapture root path and subpath into kapture into data structure.

    :param kapture_type: type of kapture data (kapture.RecordsCamera, kapture.Trajectories, ...)
    :param feature_name: type of keypoints, descriptors, global_features, used to infer the path...
    :param kapture_dirpath: root kapture path
    :return: full path of csv file for that type of data
    """
    assert kapture_type in FEATURES_CSV_FILENAMES
    filename = FEATURES_CSV_FILENAMES[kapture_type](feature_name)
    return path.join(kapture_dirpath, filename)


PADDINGS = {
    'timestamp': [8],
    'device_id': [3],
    'pose': [4, 4, 4, 4, 4, 4, 4],
}

KAPTURE_FORMAT_1 = "# kapture format: 1.1"
KAPTURE_FORMAT_PARSING_RE = '# kapture format\\:\\s*(?P<version>\\d+\\.\\d+)'

# Line separator for the kapture csv files
kapture_linesep = '\n'


def get_version_from_header(header_string: str) -> Optional[str]:
    """
    Get the kapture format version from the given header string.

    :param header_string: a header string
    :return: version as string (i.e. '2.1') if found, None otherwise
    """
    m = re.search(KAPTURE_FORMAT_PARSING_RE, header_string)
    if m:
        return m['version']
    return None


def get_version_from_csv_file(csv_file_path: str) -> Optional[str]:
    """
    Get the kapture format version from a file.

    :param csv_file_path: path to the csv file
    :return: version as string (i.e. '2.1') if found, None otherwise
    """
    if path.isfile(csv_file_path):
        with open(csv_file_path) as f:
            first_line = f.readline()
            return get_version_from_header(first_line)
    return None


def current_format_version() -> Optional[str]:
    """
    Get the current format version

    :return: format version
    """
    return get_version_from_header(KAPTURE_FORMAT_1)


def kapture_format_version(kapture_dirpath: str) -> Optional[str]:
    """
    Reads kapture format version.

    :param kapture_dirpath: kapture directory root path
    :return: kapture format version if found.
    """
    sensors_file_path = path.join(kapture_dirpath, CSV_FILENAMES[kapture.Sensors])
    return get_version_from_csv_file(sensors_file_path)


def float_safe(representation) -> Optional[float]:
    """
    Safe float cast
    https://stackoverflow.com/questions/6330071/safe-casting-in-python

    :param representation: to cast
    :return: float value if the value was castable, or None
    """
    try:
        return float(representation)
    except (ValueError, TypeError):
        return None


def float_array_or_none(representation_list) -> Optional[List[float]]:
    """
    Safe cast of list of float representations
    https://stackoverflow.com/questions/6330071/safe-casting-in-python

    :param representation_list: list of values to convert
    :return: an array of floats or None if a single one is invalid
    """
    array = [float_safe(v) for v in representation_list]
    return array if not any(v is None for v in array) else None


def table_to_file(file, table, header=None, padding=None) -> int:
    """
    Writes the given table (list of list) into a file.
            The file must be previously open as write mode.
            If table is a generator, must be valid at runtime.

    :param file: file id opened in write mode (with open(filepath, 'w') as file:)
    :param table: an iterable of iterable
    :param header: row added at the beginning of the file (+\n)
    :param padding: the padding of each column as a list of int of same size of the rows of the table.
    :return: number of records written
    """
    if header:
        file.write(KAPTURE_FORMAT_1 + kapture_linesep)
        file.write(header + kapture_linesep)
    nb_records = 0
    for row in table:
        if padding:
            row = [str(v).rjust(padding[i]) for i, v in enumerate(row)]
        file.write(', '.join(f'{v}' for v in row) + kapture_linesep)
        nb_records += 1
    return nb_records


SPLIT_PATTERN = re.compile(r'\s*,\s*')


def table_from_file(file):
    """
    Returns an iterable of iterable (generator) on the opened file.
        Be aware that the returned generator is valid as long as file is valid.

    :param file: file id opened in read mode (with open(filepath, 'r') as file:)
    :return: an iterable of iterable on kapture objects values
    """
    table = file.readlines()

    # remove end of line return, ...
    table = (line.rstrip("\n\r") for line in table)
    # remove comment lines or empty lines and trim trailing EOL
    table = (line for line in table if line.strip() and not line.startswith('#'))
    # split comma separated
    table = ([field.strip() for field in line.split(',')] for line in table)
    return list(table)


def get_last_line(opened_file: io.TextIOBase, max_line_size: int = 128) -> str:
    """
    Get the last line of an opened text file

    :param opened_file: an opened file (as returned by the builtin open)
    :param max_line_size: the maximum size of a line
    :return: last line if found, empty string otherwise
    """
    distance_to_end = 2 * max_line_size
    current_pos = opened_file.tell()
    # Check file size
    opened_file.seek(0, os.SEEK_END)
    file_size = opened_file.tell()
    if file_size > distance_to_end:
        # If we have a big file: skip towards the end
        opened_file.seek(file_size - distance_to_end, os.SEEK_SET)
    else:
        # back to start of file
        opened_file.seek(0, os.SEEK_SET)
    line = opened_file.readline()
    last_line = line
    while line:
        line = opened_file.readline()
        if line:
            last_line = line
    # back to position at the call of the function
    opened_file.seek(current_pos, os.SEEK_SET)
    return last_line


########################################################################################################################
# poses  ###############################################################################################################
def pose_to_list(pose: kapture.PoseTransform) -> List[Union[float, str]]:
    """
    Converts a 6D pose to a list of floats to save in CSV

    :param pose: 6D pose
    :return: list of float
    """
    assert (isinstance(pose, kapture.PoseTransform))
    rotation = pose.r_raw if pose.r is not None else 4 * ['']
    translation = pose.t_raw if pose.t is not None else 3 * ['']
    return rotation + translation


########################################################################################################################
# Sensor ###############################################################################################################
def sensor_to_list(sensor: kapture.Sensor) -> List[str]:
    """
    Converts a sensor into a list of strings to save in CSV

    :param sensor:
    :return: list of strings
    """
    assert (isinstance(sensor, kapture.Sensor))
    # sensor_id, name, model, [model_params]+
    fields = [sensor.name or ''] + [sensor.sensor_type] + [str(v) for v in sensor.sensor_params]
    return fields


########################################################################################################################
# Sensors ##############################################################################################################
def sensors_to_file(filepath: str, sensors: kapture.Sensors) -> None:
    """
    Writes the sensors to CSV file.

    :param filepath: input file path
    :param sensors: input sensors
    """
    assert (isinstance(sensors, kapture.Sensors))
    header = '# sensor_id, name, sensor_type, [sensor_params]+'
    table = ([sensor_id] + sensor_to_list(sensor)
             for sensor_id, sensor in sensors.items())

    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header)


def sensors_from_file(filepath: str) -> kapture.Sensors:
    """
    Reads sensors from CSV file.

    :param filepath: input file path
    :return: sensors
    """
    sensors = kapture.Sensors()
    with open(filepath) as file:
        table = table_from_file(file)
        # sensor_id, name, sensor_type, [sensor_params]+'
        for sensor_id, name, sensor_type, *sensor_params in table:
            sensor = kapture.create_sensor(sensor_type=sensor_type, sensor_params=sensor_params, name=name)
            sensors[sensor_id] = sensor

    return sensors


########################################################################################################################
# Rig ##################################################################################################################
def rigs_to_file(filepath: str, rigs: kapture.Rigs) -> None:
    """
    Writes rigs to CSV file.

    :param filepath:
    :param rigs:
    """
    assert (isinstance(rigs, kapture.Rigs))
    header = '# rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz'
    padding = PADDINGS['device_id'] + PADDINGS['device_id'] + PADDINGS['pose']
    table = ([rig_id, sensor_id] + pose_to_list(pose)
             for rig_id, rig in rigs.items()
             for sensor_id, pose in rig.items())

    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header, padding=padding)


def rigs_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None) -> kapture.Rigs:
    """
    Reads rigs from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid sensor ids.
                        If a rig id collides one of them, raise error.
                        If a sensor in rig is not in sensor_ids, it is ignored.
    :return: rigs
    """
    # rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz
    rigs = kapture.Rigs()
    with open(filepath) as file:
        table = table_from_file(file)
        for rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz in table:
            if sensor_ids is not None and rig_id in sensor_ids:
                raise ValueError(f'collision between a sensor ID and rig ID ({rig_id})')
            rotation = float_array_or_none([qw, qx, qy, qz])
            translation = float_array_or_none([tx, ty, tz])
            pose = kapture.PoseTransform(rotation, translation)
            rigs[str(rig_id), sensor_id] = pose

    if sensor_ids is not None:
        # expunge all undesired sensors
        rig_ids = set(rigs)
        for rig_id in rig_ids:
            for sensor_id in set(rigs[rig_id]):
                if sensor_id not in sensor_ids and sensor_id not in rig_ids:
                    logger.debug(f'dropping sensor {sensor_id} from rig {rig_id} because it is unknown sensor.')
                    del rigs[rig_id][sensor_id]

    return rigs


########################################################################################################################
# Trajectories #########################################################################################################
def trajectories_to_file(filepath: str, trajectories: kapture.Trajectories) -> None:
    """
    Writes trajectories to CSV file.

    :param filepath:
    :param trajectories:
    """
    assert (isinstance(trajectories, kapture.Trajectories))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz'
    padding = PADDINGS['timestamp'] + PADDINGS['device_id'] + PADDINGS['pose']
    table = (
        [timestamp, sensor_id] + pose_to_list(trajectories[(timestamp, sensor_id)])
        for timestamp, sensor_id in sorted(trajectories.key_pairs())
    )

    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header, padding=padding)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(trajectories)} in {saving_elapsed.total_seconds():.3f} seconds'
                     .replace(',', ' '))


def trajectories_from_file(filepath: str, device_ids: Optional[Set[str]] = None) -> kapture.Trajectories:
    """
    Reads trajectories from CSV file.

    :param filepath: input file path
    :param device_ids: input set of valid device ids (rig or sensor).
                        If the trajectories contains unknown devices, they will be ignored.
                        If no device_ids given, everything is loaded.
    :return: trajectories
    """
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        nb_records = 0
        trajectories = kapture.Trajectories()
        # timestamp, device_id, qw, qx, qy, qz, tx, ty, tz
        for timestamp, device_id, qw, qx, qy, qz, tx, ty, tz in table:
            if device_ids is not None and device_id not in device_ids:
                # just ignore
                continue
            pose = kapture.PoseTransform.__new__(kapture.PoseTransform)
            if qw != '' and qx != '' and qy != '' and qz != '':
                rotation = quaternion.from_float_array([float(qw), float(qx), float(qy), float(qz)])
            else:
                rotation = None
            pose._r = rotation

            if tx != '' and ty != '' and tz != '':
                trans = np.array([[float(tx)], [float(ty)], [float(tz)]], dtype=float)
            else:
                trans = None
            pose._t = trans
            trajectories.setdefault(int(timestamp), {})[device_id] = pose
            nb_records += 1
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {kapture.Trajectories} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return trajectories


########################################################################################################################
# Records Camera #######################################################################################################
def records_camera_to_file(filepath: str, records_camera: kapture.RecordsCamera) -> None:
    """
    Writes records_camera to CSV file.

    :param filepath:
    :param records_camera:
    """
    assert (isinstance(records_camera, kapture.RecordsCamera))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, image_path'
    table = (
        [timestamp, sensor_id] + [records_camera[(timestamp, sensor_id)]]
        for timestamp, sensor_id in sorted(records_camera.key_pairs())
    )
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(records_camera)} in {saving_elapsed.total_seconds():.3f} seconds'
                     .replace(',', ' '))


def records_camera_from_file(filepath: str, camera_ids: Optional[Set[str]] = None) -> kapture.RecordsCamera:
    """
    Reads records_camera from CSV file.

    :param filepath: input file path
    :param camera_ids: input set of valid camera device ids.
                        If the records_camera contains unknown devices, they will be ignored.
                        If not given, all cameras are loaded.
    :return: camera records
    """
    records_camera = kapture.RecordsCamera()
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        nb_records = 0
        # timestamp, device_id, image_path
        for timestamp, device_id, image_path in table:
            if camera_ids is not None and device_id not in camera_ids:
                # just ignore
                continue
            records_camera[(int(timestamp), str(device_id))] = image_path
            nb_records += 1
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {kapture.RecordsCamera} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return records_camera


########################################################################################################################
# Records Depth #######################################################################################################
def records_depth_to_file(filepath: str, records_depth: kapture.RecordsDepth) -> None:
    """
    Writes records_depth to CSV file.

    :param filepath:
    :param records_depth:
    """
    assert (isinstance(records_depth, kapture.RecordsDepth))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, depth_map_path'
    table = (
        [timestamp, sensor_id] + [records_depth[(timestamp, sensor_id)]]
        for timestamp, sensor_id in sorted(records_depth.key_pairs())
    )
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(records_depth)} in {saving_elapsed.total_seconds():.3f} seconds'
                     .replace(',', ' '))


def records_depth_from_file(filepath: str, camera_ids: Optional[Set[str]] = None) -> kapture.RecordsDepth:
    """
    Reads records_depth from CSV file.

    :param filepath: input file path
    :param camera_ids: input set of valid camera device ids.
                        If the records_camera contains unknown devices, they will be ignored.
                        If not given, all cameras are loaded.
    :return: camera records
    """
    records_depth = kapture.RecordsDepth()
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        nb_records = 0
        # timestamp, device_id, image_path
        for timestamp, device_id, depth_map_path in table:
            if camera_ids is not None and device_id not in camera_ids:
                # just ignore
                continue
            records_depth[(int(timestamp), str(device_id))] = depth_map_path
            nb_records += 1
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {kapture.RecordsDepth} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return records_depth


########################################################################################################################
# Records Lidar ########################################################################################################
def records_lidar_to_file(filepath: str, records_lidar: kapture.RecordsLidar) -> None:
    """
    Writes records_lidar to CSV file.

    :param filepath:
    :param records_lidar:
    """
    assert (isinstance(records_lidar, kapture.RecordsLidar))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, point_cloud_path'
    table = (
        [timestamp, sensor_id] + [records_lidar[(timestamp, sensor_id)]]
        for timestamp, sensor_id in sorted(records_lidar.key_pairs())
    )
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(records_lidar)} in {saving_elapsed.total_seconds():.3f} seconds'
                     .replace(',', ' '))


def records_lidar_from_file(filepath: str, lidar_ids: Optional[Set[str]] = None
                            ) -> kapture.RecordsLidar:
    """
    Reads records_lidar from CSV file.

    :param filepath: input file path
    :param lidar_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                            If omitted, then it loads all devices.
    :return: Lidar records
    """
    records_lidar = kapture.RecordsLidar()
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        nb_records = 0
        # timestamp, device_id, point_cloud_path
        for timestamp, device_id, point_cloud_path in table:
            if lidar_ids is not None and device_id not in lidar_ids:
                # just ignore
                continue
            records_lidar[(int(timestamp), str(device_id))] = point_cloud_path
            nb_records += 1
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {kapture.RecordsLidar} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return records_lidar


########################################################################################################################
def records_generic_to_file(filepath: str, records: kapture.RecordsBase) -> None:
    """
        Writes records_wifi to file

        :param filepath: path where to save records file.
        :param records:
    """
    assert (isinstance(records, kapture.RecordsBase))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, ' + ', '.join(f.name for f in records.record_type.fields())
    table = []
    for timestamp, sensor_id, record in kapture.flatten(records, is_sorted=True):
        table.append([timestamp, sensor_id] + [str(v) for v in record.astuple()])
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(records)} in {saving_elapsed.total_seconds():.3f} seconds'
                     .replace(',', ' '))


def records_generic_from_file(records_type: Type, filepath: str, sensor_ids: Optional[Set[str]] = None
                              ) -> Union[kapture.RecordsBase,
                                         kapture.RecordsGnss,
                                         kapture.RecordsGyroscope,
                                         kapture.RecordsAccelerometer,
                                         kapture.RecordsMagnetic]:
    """
    Reads Records data from CSV file.

    :param records_type: type of records expected (eg RecordsWifi)
    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                     If omitted, then it loads all devices.
    :return: records
    """
    records = records_type()
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        # timestamp, device_id, *
        nb_records = 0
        for timestamp, device_id, *data in table:
            timestamp = int(timestamp)
            device_id = str(device_id)
            if sensor_ids is not None and device_id not in sensor_ids:
                # just ignore
                continue
            records[timestamp, device_id] = records_type.record_type(*data)
            nb_records += 1

    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {records_type} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return records


# Records Wifi #########################################################################################################
def records_wifi_to_file(filepath: str, records_wifi: kapture.RecordsWifi) -> None:
    """
    Writes records_wifi to file

    :param filepath:
    :param records_wifi:
    """
    assert (isinstance(records_wifi, kapture.RecordsWifi))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, BSSID, frequency, RSSI, SSID, scan_time_start, scan_time_end'
    table = []
    for timestamp, sensor_id in sorted(records_wifi.key_pairs()):
        for bssid, record in records_wifi[timestamp, sensor_id].items():
            table.append([timestamp, sensor_id, bssid] + [str(v) for v in record.astuple()])
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(records_wifi)} in {saving_elapsed.total_seconds():.3f} seconds'
                     .replace(',', ' '))


def records_wifi_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None
                           ) -> kapture.RecordsWifi:
    """
    Reads RecordsWifi from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                            If omitted, then it loads all devices.
    :return: Wifi records
    """
    records_wifi = kapture.RecordsWifi()
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        nb_records = 0
        # timestamp, device_id, BSSID, frequency, RSSI, SSID, scan_time_start, scan_time_end
        for timestamp, device_id, BSSID, frequency, RSSI, SSID, scan_time_start, scan_time_end in table:
            timestamp, device_id = int(timestamp), str(device_id)
            if sensor_ids is not None and device_id not in sensor_ids:
                # just ignore
                continue
            if (timestamp, device_id) not in records_wifi:
                records_wifi[timestamp, device_id] = kapture.RecordWifi()
            records_wifi[timestamp, device_id][BSSID] = kapture.RecordWifiSignal(
                frequency, RSSI, SSID, scan_time_start, scan_time_end)
            nb_records += 1

    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {kapture.RecordsWifi} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return records_wifi


# Records Bluetooth ####################################################################################################
def records_bluetooth_to_file(filepath: str, records_bluetooth: kapture.RecordsBluetooth) -> None:
    """
    Writes Bluetooth records to file

    :param filepath: output file path.
    :param records_bluetooth: records to save
    """
    assert (isinstance(records_bluetooth, kapture.RecordsBluetooth))
    saving_start = datetime.datetime.now()
    header = '# timestamp, device_id, address, RSSI, name'
    table = []
    for timestamp, sensor_id in sorted(records_bluetooth.key_pairs()):
        for address, bt_record in records_bluetooth[timestamp, sensor_id].items():
            table.append([timestamp, sensor_id, address] + [str(v) for v in bt_record.astuple()])
    with open(filepath, 'w') as file:
        nb_records = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_records:12,d} {type(records_bluetooth)}'
                     f' in {saving_elapsed.total_seconds():.3f} seconds'.replace(',', ' '))


def records_bluetooth_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None
                                ) -> kapture.RecordsBluetooth:
    """
    Reads Bluetooth records from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                        If omitted, then it loads all devices.
    :return: Bluetooth records
    """
    records_bluetooth = kapture.RecordsBluetooth()
    loading_start = datetime.datetime.now()
    with open(filepath) as file:
        table = table_from_file(file)
        nb_records = 0
        # timestamp, device_id, address, RSSI, name
        for timestamp, device_id, address, RSSI, name in table:
            timestamp, device_id = int(timestamp), str(device_id)
            if sensor_ids is not None and device_id not in sensor_ids:
                # just ignore
                continue
            if (timestamp, device_id) not in records_bluetooth:
                records_bluetooth[timestamp, device_id] = kapture.RecordBluetooth()
            records_bluetooth[timestamp, device_id][address] = kapture.RecordBluetoothSignal(rssi=RSSI, name=name)
            nb_records += 1

    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{nb_records:12,d} {kapture.RecordsBluetooth} in {loading_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))
    return records_bluetooth


# GNSS #################################################################################################################
def records_gnss_to_file(filepath: str, records_gnss: kapture.RecordsGnss) -> None:
    """
    Writes Gnss records to file

    :param filepath: output file path.
    :param records_gnss: records to save
    """
    records_generic_to_file(filepath, records_gnss)


def records_gnss_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None
                           ) -> kapture.RecordsGnss:
    """
    Reads Gnss records from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                        If omitted, then it loads all devices.
    :return: Gnss records
    """
    return records_generic_from_file(kapture.RecordsGnss, filepath, sensor_ids)


# Accelerometer ########################################################################################################
def records_accelerometer_to_file(filepath: str, records_accelerometer: kapture.RecordsAccelerometer) -> None:
    """
    Writes accelerometer records to file

    :param filepath: output file path.
    :param records_accelerometer: records to save
    """
    records_generic_to_file(filepath, records_accelerometer)


def records_accelerometer_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None
                                    ) -> kapture.RecordsAccelerometer:
    """
    Reads accelerometer records from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                        If omitted, then it loads all devices.
    :return: accelerometer records
    """
    return records_generic_from_file(kapture.RecordsAccelerometer, filepath, sensor_ids)


# Gyroscope ########################################################################################################
def records_gyroscope_to_file(filepath: str, records_gyroscope: kapture.RecordsGyroscope) -> None:
    """
    Writes gyroscope records to file

    :param filepath: output file path.
    :param records_gyroscope: records to save
    """
    records_generic_to_file(filepath, records_gyroscope)


def records_gyroscope_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None
                                ) -> kapture.RecordsGyroscope:
    """
    Reads gyroscope records from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                        If omitted, then it loads all devices.
    :return: gyroscope records
    """
    return records_generic_from_file(kapture.RecordsGyroscope, filepath, sensor_ids)


# Magnetic ########################################################################################################
def records_magnetic_to_file(filepath: str, records_magnetic: kapture.RecordsMagnetic) -> None:
    """
    Writes magnetic records to file

    :param filepath: output file path.
    :param records_magnetic: records to save
    """
    records_generic_to_file(filepath, records_magnetic)


def records_magnetic_from_file(filepath: str, sensor_ids: Optional[Set[str]] = None
                               ) -> kapture.RecordsMagnetic:
    """
    Reads magnetic records from CSV file.

    :param filepath: input file path
    :param sensor_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                        If omitted, then it loads all devices.
    :return: magnetic records
    """
    return records_generic_from_file(kapture.RecordsMagnetic, filepath, sensor_ids)


########################################################################################################################
# features #############################################################################################################
def image_features_set_from_dir(
        kapture_type: Type[Union[kapture.Keypoints,
                                 kapture.Descriptors,
                                 kapture.GlobalFeatures]],
        feature_type: str,
        kapture_dirpath: str,
        image_filenames: Optional[Set[str]]
) -> Set[str]:
    """
    Reads and builds ImageFeatures from images_filenames if given, or directly from actual files in  root_dirpath.

    :param kapture_type: kapture class type.
    :param feature_type: name of the feature
    :param kapture_dirpath: input path to kapture root directory.
    :param image_filenames: None or set of image relative paths.
    :return: Set
    """

    if image_filenames is None:
        # images_path is empty, so populates all feature files
        image_filenames_generator = kapture.io.features.image_ids_from_feature_dirpath(kapture_type,
                                                                                       feature_type,
                                                                                       kapture_dirpath)
    else:
        # filter only existing files
        image_filenames_generator = (
            image_name
            for image_name in image_filenames
            if path.exists(kapture.io.features.get_features_fullpath(kapture_type,
                                                                     feature_type,
                                                                     kapture_dirpath,
                                                                     image_name)))
    return set(image_filenames_generator)


def image_features_set_from_tar(
        kapture_type: Type[Union[kapture.Keypoints,
                                 kapture.Descriptors,
                                 kapture.GlobalFeatures]],
        tar_handler: TarHandler,
        image_filenames: Optional[Set[str]]
) -> Set[str]:
    """
    Reads and builds ImageFeatures from images_filenames if given, or directly from actual files in the tar.

    :param kapture_type: kapture class type.
    :param tar_handler: opened tar reference
    :param image_filenames: None or set of image relative paths.
    :return: Set
    """
    image_filenames_generator = kapture.io.features.image_ids_from_feature_tar(kapture_type, tar_handler)
    if image_filenames is not None:
        image_filenames_generator = (image_name
                                     for image_name in image_filenames_generator
                                     if image_name in image_filenames)
    return set(image_filenames_generator)


########################################################################################################################
# keypoints ############################################################################################################
KeypointsConfig = namedtuple('KeypointsConfig', ['name', 'dtype', 'dsize'])


def keypoints_to_file(config_filepath: str, keypoints: kapture.Keypoints) -> None:
    """
    Writes keypoints to CSV file.

    :param config_filepath:
    :param keypoints:
    """
    os.makedirs(path.dirname(config_filepath), exist_ok=True)
    header = "# name, dtype, dsize"
    dtype = str(keypoints.dtype) if isinstance(keypoints.dtype, np.dtype) else keypoints.dtype.__name__
    line = [keypoints.type_name, dtype, str(keypoints.dsize)]
    with open(config_filepath, 'wt') as file:
        table_to_file(file, [line], header=header)


def keypoints_config_from_file(config_filepath: str) -> KeypointsConfig:
    """
    Reads keypoints config files, ie. name, data type, and data size.
    :param config_filepath: input path to config file to read.
    :return: config The read config.
    """
    if not path.exists(config_filepath):
        raise FileNotFoundError(f'{path.basename(config_filepath)} file is missing')

    with open(config_filepath, 'rt') as file:
        table = table_from_file(file)
        line = list(table)[0]
        assert len(line) == 3
        name, dtype, dsize = line[0], line[1], int(line[2])

    # try to list all possible type from numpy that can be used in eval(dtype)
    from numpy import float, float32, float64, int32, uint8  # noqa: F401
    if isinstance(type(eval(dtype)), type):
        dtype = eval(dtype)
    else:
        raise ValueError('Expect data type ')
    return KeypointsConfig(name, dtype, dsize)


def keypoints_from_dir(keypoints_type: str,
                       kapture_dirpath: str,
                       images_paths: Optional[Set[str]],
                       tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> kapture.Keypoints:
    """
    Reads and builds keypoints from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: root path of kapture
    :param images_paths: optional list of image file names
    :param tar_handler: collection of preloaded tar archives
    :return: Keypoints
    """
    # make config_filepath from data_dirpath
    config_filepath = get_feature_csv_fullpath(kapture.Keypoints, keypoints_type, kapture_dirpath)
    config = keypoints_config_from_file(config_filepath)

    # exist as tar ?
    tar_local_handler = retrieve_tar_handler_from_collection(kapture.Keypoints, keypoints_type, tar_handler)

    if tar_local_handler is not None:
        image_filenames = image_features_set_from_tar(kapture.Keypoints, tar_local_handler, images_paths)
    else:
        # exist tar ? -> fire warning
        tar_path = get_feature_tar_fullpath(kapture.Keypoints, keypoints_type, kapture_dirpath)
        if os.path.isfile(tar_path):
            getLogger().warning(f'{tar_path} exist but no handler was given so it is ignored and loaded from dir')
        image_filenames = image_features_set_from_dir(kapture.Keypoints,
                                                      keypoints_type,
                                                      kapture_dirpath,
                                                      images_paths)
    return kapture.Keypoints(config.name, config.dtype, config.dsize, image_filenames)


########################################################################################################################
# descriptors ##########################################################################################################
DescriptorsConfig = namedtuple('DescriptorsConfig', ['name', 'dtype', 'dsize', 'keypoints_type', 'metric_type'])


def descriptors_to_file(config_filepath: str, descriptors: kapture.Descriptors) -> None:
    """
    Writes descriptors to CSV file.
    :param config_filepath:
    :param descriptors:
    """
    os.makedirs(path.dirname(config_filepath), exist_ok=True)
    header = "# name, dtype, dsize, keypoints_type, metric_type"
    dtype = str(descriptors.dtype) if isinstance(descriptors.dtype, np.dtype) else descriptors.dtype.__name__
    line = [descriptors.type_name, dtype, str(descriptors.dsize), descriptors.keypoints_type, descriptors.metric_type]
    with open(config_filepath, 'wt') as file:
        table_to_file(file, [line], header=header)


def descriptors_config_from_file(config_filepath: str) -> DescriptorsConfig:
    """
    Reads descriptors config files.
    :param config_filepath: input path to config file to read.
    :return: config The read config.
    """
    if not path.exists(config_filepath):
        raise FileNotFoundError(f'{path.basename(config_filepath)} file is missing')

    with open(config_filepath, 'rt') as file:
        table = table_from_file(file)
        line = list(table)[0]
        assert len(line) == 5
        name, dtype, dsize, keypoints_type, metric_type = line[0], line[1], int(line[2]), line[3], line[4]

    # try to list all possible type from numpy that can be used in eval(dtype)
    from numpy import float, float32, float64, int32, uint8  # noqa: F401
    if isinstance(type(eval(dtype)), type):
        dtype = eval(dtype)
    else:
        raise ValueError('Expect data type ')
    return DescriptorsConfig(name, dtype, dsize, keypoints_type, metric_type)


def descriptors_from_dir(descriptors_type: str, kapture_dirpath: str, images_paths: Optional[Set[str]],
                         tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> kapture.Descriptors:
    """
    Reads and builds descriptors from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param descriptors_type: type of descriptors to export, name of the descriptors subfolder
    :param kapture_dirpath: root path of kapture
    :param images_paths: optional list of image file names
    :param tar_handler: collection of preloaded tar archives
    :return: Descriptors
    """
    # make config_filepath from data_dirpath
    config_filepath = get_feature_csv_fullpath(kapture.Descriptors, descriptors_type, kapture_dirpath)
    config = descriptors_config_from_file(config_filepath)

    # exist as tar ?
    tar_local_handler = retrieve_tar_handler_from_collection(kapture.Descriptors, descriptors_type, tar_handler)

    if tar_local_handler is not None:
        image_filenames = image_features_set_from_tar(kapture.Descriptors, tar_local_handler, images_paths)
    else:
        # exist tar ? -> fire warning
        tar_path = get_feature_tar_fullpath(kapture.Descriptors, descriptors_type, kapture_dirpath)
        if os.path.isfile(tar_path):
            getLogger().warning(f'{tar_path} exist but no handler was given so it is ignored and loaded from dir')
        image_filenames = image_features_set_from_dir(kapture.Descriptors,
                                                      descriptors_type,
                                                      kapture_dirpath,
                                                      images_paths)
    return kapture.Descriptors(config.name,
                               config.dtype,
                               config.dsize,
                               config.keypoints_type,
                               config.metric_type,
                               image_filenames)


########################################################################################################################
# global_features ######################################################################################################
GlobalFeaturesConfig = namedtuple('GlobalFeaturesConfig', ['name', 'dtype', 'dsize', 'metric_type'])


def global_features_to_file(config_filepath: str, global_features: kapture.GlobalFeatures) -> None:
    """
    Writes global features to CSV file.

    :param config_filepath:
    :param global_features:
    """
    os.makedirs(path.dirname(config_filepath), exist_ok=True)
    header = "# name, dtype, dsize, metric_type"
    if isinstance(global_features.dtype, np.dtype):
        dtype = str(global_features.dtype)
    else:
        dtype = global_features.dtype.__name__
    line = [global_features.type_name, dtype, str(global_features.dsize), global_features.metric_type]
    with open(config_filepath, 'wt') as file:
        table_to_file(file, [line], header=header)


def global_features_config_from_file(config_filepath: str) -> GlobalFeaturesConfig:
    """
    Reads global_features config files.
    :param config_filepath: input path to config file to read.
    :return: config The read config.
    """
    if not path.exists(config_filepath):
        raise FileNotFoundError(f'{path.basename(config_filepath)} file is missing')

    with open(config_filepath, 'rt') as file:
        table = table_from_file(file)
        line = list(table)[0]
        assert len(line) == 4
        name, dtype, dsize, metric_type = line[0], line[1], int(line[2]), line[3]

    # try to list all possible type from numpy that can be used in eval(dtype)
    from numpy import float, float32, float64, int32, uint8  # noqa: F401
    if isinstance(type(eval(dtype)), type):
        dtype = eval(dtype)
    else:
        raise ValueError('Expect data type ')
    return GlobalFeaturesConfig(name, dtype, dsize, metric_type)


def global_features_from_dir(global_features_type: str,
                             kapture_dirpath: str,
                             images_paths: Set[str],
                             tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> kapture.GlobalFeatures:
    """
    Reads and builds Global features from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param global_features_type: type of global_features, name of the global_features subfolder
    :param kapture_dirpath: root path of kapture
    :param images_paths: optional list of image file names
    :param tar_handler: collection of preloaded tar archives
    :return: Global features
    """
    # make config_filepath from data_dirpath
    config_filepath = get_feature_csv_fullpath(kapture.GlobalFeatures, global_features_type, kapture_dirpath)
    config = global_features_config_from_file(config_filepath)

    # exist as tar ?
    tar_local_handler = retrieve_tar_handler_from_collection(kapture.GlobalFeatures, global_features_type, tar_handler)

    if tar_local_handler is not None:
        image_filenames = image_features_set_from_tar(kapture.GlobalFeatures, tar_local_handler, images_paths)
    else:
        # exist tar ? -> fire warning
        tar_path = get_feature_tar_fullpath(kapture.GlobalFeatures, global_features_type, kapture_dirpath)
        if os.path.isfile(tar_path):
            getLogger().warning(f'{tar_path} exist but no handler was given so it is ignored and loaded from dir')
        image_filenames = image_features_set_from_dir(kapture.GlobalFeatures,
                                                      global_features_type,
                                                      kapture_dirpath,
                                                      images_paths)
    return kapture.GlobalFeatures(config.name,
                                  config.dtype,
                                  config.dsize,
                                  config.metric_type,
                                  image_filenames)


########################################################################################################################
# matches ##############################################################################################################
def matches_from_dir(
        keypoints_type: str,
        kapture_dirpath: str,
        image_filenames: Optional[Set[str]] = None,
        matches_pairsfile_path: Optional[str] = None,
        tar_handler: Optional[Union[TarCollection, TarHandler]] = None) -> kapture.Matches:
    """
    Reads and builds Matches from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param keypoints_type: type of keypoints, name of the keypoints subfolder
    :param kapture_dirpath: root path of kapture
    :param image_filenames: optional list of image file names
    :param matches_pairsfile_path: text file in the csv format; where each line is image_name1, image_name2, score
    :param tar_handler: collection of preloaded tar archives
    :return: Matches
    """
    # exist as tar ?
    tar_local_handler = retrieve_tar_handler_from_collection(kapture.Matches, keypoints_type, tar_handler)
    loading_start = datetime.datetime.now()
    if tar_local_handler is not None:
        match_pairs_generator = kapture.io.features.matching_pairs_from_tar(tar_local_handler)
        if matches_pairsfile_path is not None:
            with open(matches_pairsfile_path, 'r') as fid:
                table = table_from_file(fid)
                valid_pairs = set((query_name, map_name) if query_name < map_name else (map_name, query_name)
                                  for query_name, map_name, _ in table)
                match_pairs_generator = (image_pair
                                         for image_pair in match_pairs_generator
                                         if image_pair in valid_pairs)
    else:
        # exist tar ? -> fire warning
        tar_path = get_feature_tar_fullpath(kapture.Matches, keypoints_type, kapture_dirpath)
        if os.path.isfile(tar_path):
            getLogger().warning(f'{tar_path} exist but no handler was given so it is ignored and loaded from dir')
        if matches_pairsfile_path is None:
            # populate files from disk
            match_pairs_generator = kapture.io.features.matching_pairs_from_dirpath(keypoints_type, kapture_dirpath)
        else:
            with open(matches_pairsfile_path, 'r') as fid:
                table = table_from_file(fid)
                # get matches list from pairsfile
                match_pairs_generator = ((query_name, map_name) if query_name < map_name else (map_name, query_name)
                                         for query_name, map_name, _ in table)
                # keeps only the one that actually exists on disk
                match_pairs_generator = (image_pair
                                         for image_pair in match_pairs_generator
                                         if path.isfile(kapture.io.features.get_matches_fullpath(image_pair,
                                                                                                 keypoints_type,
                                                                                                 kapture_dirpath))
                                         )
    if image_filenames is not None:
        # retains only files that correspond to known images
        match_pairs_generator = (
            image_pair
            for image_pair in match_pairs_generator
            if image_pair[0] in image_filenames and image_pair[1] in image_filenames
        )
    match_pairs = set(match_pairs_generator)
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{len(match_pairs):12,d} {kapture.Matches} in {loading_elapsed.total_seconds():.3f} seconds')
    return kapture.Matches(match_pairs)


########################################################################################################################
# points3d #############################################################################################################
def points3d_to_file(filepath: str, points3d: kapture.Points3d) -> None:
    """
    Writes 3d points to CSV file.

    :param filepath: path to CSV file
    :param points3d: the 3d points
    """
    assert isinstance(points3d, kapture.Points3d)
    os.makedirs(path.dirname(filepath), exist_ok=True)
    saving_start = datetime.datetime.now()
    header = KAPTURE_FORMAT_1[2:] + kapture_linesep + 'X, Y, Z, R, G, B'
    np.savetxt(filepath, points3d.as_array(), delimiter=',', header=header, fmt='%.10f')
    saving_elapsed = datetime.datetime.now() - saving_start
    logger.debug(f'wrote {len(points3d):12,d} {type(points3d)} in {saving_elapsed.total_seconds():.3f} seconds'
                 .replace(',', ' '))


def points3d_from_file(filepath: str) -> kapture.Points3d:
    """
    Reads 3d points from CSV file.

    :param filepath: path to CSV file
    :return: the 3d points
    """

    loading_start = datetime.datetime.now()
    data = np.loadtxt(filepath, dtype=np.float, delimiter=',', comments='#')
    data = data.reshape((-1, kapture.Points3d.XYZRGB))  # make sure of the shape, even if single line file.
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{len(data):12,d} {kapture.Points3d} in {loading_elapsed.total_seconds():.3f} seconds')
    return kapture.Points3d(data)


########################################################################################################################
# observations #########################################################################################################
def observations_to_file(observations_filepath: str, observations: kapture.Observations) -> None:
    """
    Writes observations to CSV file.

    :param observations_filepath: input path to CSV file of observation to write.
                                    Containing directory is created if needed.
    :param observations: input observations to be written.
    """
    assert path.basename(observations_filepath) == path.basename(CSV_FILENAMES[kapture.Observations])
    assert isinstance(observations, kapture.Observations)
    saving_start = datetime.datetime.now()
    header = '# point3d_id, keypoints_type, [image_path, feature_id]*'
    table = (
        [str(point3d_idx), str(keypoints_type)] + [str(k)
                                                   for pair in observations[point3d_idx, keypoints_type]
                                                   for k in pair]
        for point3d_idx, keypoints_type in sorted(observations.key_pairs(), key=lambda x: (x[0], x[1]))
    )
    os.makedirs(path.dirname(observations_filepath), exist_ok=True)
    with open(observations_filepath, 'w') as file:
        nb_lines = table_to_file(file, table, header=header)
        saving_elapsed = datetime.datetime.now() - saving_start
        logger.debug(f'wrote {nb_lines:12,d} lines with {observations.observations_number()} {type(observations)}'
                     f' in {saving_elapsed.total_seconds():.3f} seconds'.replace(',', ' '))


def observations_from_file(observations_filepath: str, loaded_keypoints: Optional[Dict[str, Set[str]]] = None)\
        -> kapture.Observations:
    """
    Reads observations from CSV file.

    :param observations_filepath: path to CSV file to read.
    :param loaded_keypoints: input set of image names (ids) that have keypoints.
                                        If given, used to filter out irrelevant observations.
                                        You can get from set(kapture.keypoints)
    :return: observations
    """
    assert path.basename(observations_filepath) == path.basename(CSV_FILENAMES[kapture.Observations])
    assert loaded_keypoints is None \
        or (isinstance(loaded_keypoints, dict) and len(loaded_keypoints) > 0)
    assert loaded_keypoints is None or all([isinstance(keypoints, set) for keypoints in loaded_keypoints.values()])

    observations = kapture.Observations()
    loading_start = datetime.datetime.now()
    with open(observations_filepath) as file:
        table = table_from_file(file)
        nb_observations = 0
        # point3d_id, keypoints_type, [image_path, feature_id]*
        for points3d_id_str, keypoints_type, *pairs in table:
            if loaded_keypoints is not None and \
                    (keypoints_type not in loaded_keypoints or len(loaded_keypoints[keypoints_type]) == 0):
                continue
            points3d_id = int(points3d_id_str)
            if len(pairs) > 1:
                image_paths = pairs[0::2]
                keypoints_ids = pairs[1::2]
                for image_path, keypoint_id in zip(image_paths, keypoints_ids):
                    if loaded_keypoints is not None and image_path not in loaded_keypoints[keypoints_type]:
                        # image_path does not exist in kapture (perhaps it was removed), ignore it
                        continue
                    observations.add(points3d_id, keypoints_type, image_path, int(keypoint_id))
                nb_observations += int(len(pairs)/2)
    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'{len(table):12,d} lines with {nb_observations} {kapture.Observations}'
                 f' in {loading_elapsed.total_seconds():.3f} seconds'.replace(',', ' '))
    return observations


########################################################################################################################
# Kapture Write ########################################################################################################
KAPTURE_ATTRIBUTE_WRITERS = {
    kapture.Sensors: sensors_to_file,
    kapture.Rigs: rigs_to_file,
    kapture.Trajectories: trajectories_to_file,
    kapture.RecordsCamera: records_camera_to_file,
    kapture.RecordsDepth: records_depth_to_file,
    kapture.RecordsLidar: records_lidar_to_file,
    kapture.RecordsWifi: records_wifi_to_file,
    kapture.RecordsBluetooth: records_bluetooth_to_file,
    kapture.RecordsGnss: records_gnss_to_file,
    kapture.RecordsAccelerometer: records_accelerometer_to_file,
    kapture.RecordsGyroscope: records_gyroscope_to_file,
    kapture.RecordsMagnetic: records_magnetic_to_file,
    kapture.Keypoints: keypoints_to_file,
    kapture.Descriptors: descriptors_to_file,
    kapture.GlobalFeatures: global_features_to_file,
    kapture.Points3d: points3d_to_file,
    kapture.Observations: observations_to_file,
}

KAPTURE_ATTRIBUTE_NAMES = {  # used to list attributes to be saved
    kapture.Sensors: 'sensors',
    kapture.Rigs: 'rigs',
    kapture.Trajectories: 'trajectories',
    kapture.RecordsCamera: 'records_camera',
    kapture.RecordsDepth: 'records_depth',
    kapture.RecordsLidar: 'records_lidar',
    kapture.RecordsWifi: 'records_wifi',
    kapture.RecordsBluetooth: 'records_bluetooth',
    kapture.RecordsGnss: 'records_gnss',
    kapture.RecordsAccelerometer: 'records_accelerometer',
    kapture.RecordsGyroscope: 'records_gyroscope',
    kapture.RecordsMagnetic: 'records_magnetic',
    kapture.Keypoints: 'keypoints',
    kapture.Descriptors: 'descriptors',
    kapture.GlobalFeatures: 'global_features',
    kapture.Points3d: 'points3d',
    kapture.Observations: 'observations',
}


def kapture_to_dir(kapture_dirpath: str, kapture_data: kapture.Kapture) -> None:
    """
    Saves kapture data to given directory.

    :param kapture_dirpath: kapture directory root path
    :param kapture_data: input kapture data
    """
    kapture_subtype_to_filepaths = {kapture_class: path.join(kapture_dirpath, filename)
                                    for kapture_class, filename in CSV_FILENAMES.items()}
    saving_start = datetime.datetime.now()
    # save each member of kapture data
    for kapture_class, kapture_member_name in KAPTURE_ATTRIBUTE_NAMES.items():
        part_data = kapture_data.__getattribute__(kapture_member_name)
        if part_data is not None and kapture_class in kapture_subtype_to_filepaths:
            # save it
            logger.debug(f'saving {kapture_member_name} ...')
            write_function = KAPTURE_ATTRIBUTE_WRITERS[kapture_class]
            write_function(kapture_subtype_to_filepaths[kapture_class], part_data)
        elif part_data is not None and kapture_class in FEATURES_CSV_FILENAMES:
            write_function = KAPTURE_ATTRIBUTE_WRITERS[kapture_class]
            for feature_type, features in part_data.items():
                # save it
                logger.debug(f'saving {kapture_member_name} : {feature_type} ...')
                filepath = path.join(kapture_dirpath, FEATURES_CSV_FILENAMES[kapture_class](feature_type))
                write_function(filepath, features)
    saving_elapsed = datetime.datetime.now() - saving_start
    logger.info(f'Saved in {saving_elapsed.total_seconds():.3f} seconds in "{kapture_dirpath}"')


# Kapture Read #########################################################################################################
# list all data members of kapture.
KAPTURE_LOADABLE_TYPES = {
    kapture.Sensors,
    kapture.Rigs,
    kapture.Trajectories,
    kapture.RecordsCamera,
    kapture.RecordsDepth,
    kapture.RecordsLidar,
    kapture.RecordsWifi,
    kapture.RecordsBluetooth,
    kapture.RecordsGnss,
    kapture.RecordsAccelerometer,
    kapture.RecordsGyroscope,
    kapture.RecordsMagnetic,
    kapture.Keypoints,
    kapture.Descriptors,
    kapture.GlobalFeatures,
    kapture.Matches,
    kapture.Points3d,
    kapture.Observations,
}


def kapture_from_dir(
        kapture_dir_path: str,
        matches_pairs_file_path: Optional[str] = None,
        skip_list: List[Type[Union[kapture.Rigs,
                                   kapture.Trajectories,
                                   kapture.RecordsCamera,
                                   kapture.RecordsDepth,
                                   kapture.RecordsLidar,
                                   kapture.RecordsWifi,
                                   kapture.RecordsBluetooth,
                                   kapture.RecordsGnss,
                                   kapture.RecordsAccelerometer,
                                   kapture.RecordsGyroscope,
                                   kapture.RecordsMagnetic,
                                   kapture.Keypoints,
                                   kapture.Descriptors,
                                   kapture.GlobalFeatures,
                                   kapture.Matches,
                                   kapture.Points3d,
                                   kapture.Observations
                                   ]]] = [],
        tar_handlers: Optional[TarCollection] = None
) -> kapture.Kapture:
    """
    Reads and return kapture data from directory.

    :param kapture_dir_path: kapture directory root path
    :param matches_pairs_file_path: text file in the csv format; where each line is image_name1, image_name2, score
    :param skip_list: Input option for expert only. Skip the load of specified parts.
    :param tar_handlers: collection of preloaded tar archives
    :return: kapture data read
    """
    if not path.isdir(kapture_dir_path):
        raise FileNotFoundError(f'No kapture directory {kapture_dir_path}')
    csv_file_paths = {dtype: path.join(kapture_dir_path, filename)
                      for dtype, filename in CSV_FILENAMES.items()}
    data_dir_paths = {dtype: path.join(kapture_dir_path, dir_name)
                      for dtype, dir_name in kapture.io.features.FEATURES_DATA_DIRNAMES.items()}

    # keep only those in load_only and that exists
    kapture_data_paths = {**data_dir_paths, **csv_file_paths}  # make sure files take precedence over dirs
    kapture_loadable_data = {
        kapture_type
        for kapture_type in KAPTURE_LOADABLE_TYPES
        if kapture_type not in skip_list and path.exists(kapture_data_paths[kapture_type])
    }

    kapture_data = kapture.Kapture()
    loading_start = datetime.datetime.now()
    # sensors
    sensor_ids = None
    sensors_file_path = csv_file_paths[kapture.Sensors]
    if sensors_file_path:
        logger.debug(f'loading sensors {sensors_file_path} ...')
        assert path.isfile(sensors_file_path)
        kapture_data.__version__ = get_version_from_csv_file(sensors_file_path)
        try:
            if float(kapture_data.__version__) > float(current_format_version()):
                raise ValueError(f'unable to load version over {current_format_version()}')
        except ValueError:
            raise FileNotFoundError(f'unable to load version over {current_format_version()}')

        kapture_data.sensors = sensors_from_file(sensors_file_path)
        sensor_ids = set(kapture_data.sensors.keys()) if kapture_data.sensors is not None else set()

    if sensor_ids is None:
        # no need to continue, everything else depends on sensors
        raise FileNotFoundError(f'File {sensors_file_path} is missing or empty in {kapture_dir_path}')

    # rigs
    if kapture.Rigs in kapture_loadable_data:
        rigs_file_path = csv_file_paths[kapture.Rigs]
        logger.debug(f'loading rigs {rigs_file_path} ...')
        assert sensor_ids is not None
        kapture_data.rigs = rigs_from_file(rigs_file_path, sensor_ids)
        # update sensor_ids with rig_id
        sensor_ids.update(kapture_data.rigs.keys())

    # trajectories
    if kapture.Trajectories in kapture_loadable_data:
        trajectories_file_path = csv_file_paths[kapture.Trajectories]
        logger.debug(f'loading trajectories {trajectories_file_path} ...')
        assert sensor_ids is not None
        kapture_data.trajectories = trajectories_from_file(trajectories_file_path, sensor_ids)

    _load_all_records(csv_file_paths, kapture_loadable_data, kapture_data)
    # be picky on version number for desc, feat and matches
    if kapture_data.__version__ and kapture_data.__version__ == current_format_version():
        _load_features_and_desc_and_matches(data_dir_paths, kapture_dir_path, matches_pairs_file_path,
                                            kapture_loadable_data, kapture_data, tar_handlers)
        _load_points3d_and_observations(csv_file_paths, kapture_loadable_data, kapture_data)
    else:
        logger.critical(f'unsupported version ({kapture_data.__version__}): skip loading reconstruction part. '
                        f'Please upgrade to {current_format_version()}.')

    loading_elapsed = datetime.datetime.now() - loading_start
    logger.debug(f'Loaded in {loading_elapsed.total_seconds():.3f} seconds from "{kapture_dir_path}"')
    return kapture_data


def get_sensor_ids_of_type(sensor_type: str, sensors: kapture.Sensors) -> Set[str]:
    """
    Get the sensors of a certain kapture type ('camera', 'lidar', ...)

    :param sensor_type: type of sensor
    :param sensors: sensors to process
    :return: sensors identifiers
    """
    return set([sensor_id
                for sensor_id in sensors.keys()
                if sensors[sensor_id].sensor_type == sensor_type])


def _load_all_records(csv_file_paths, kapture_loadable_data, kapture_data) -> None:
    """
    Loads all records from disk to the kapture in memory

    :param csv_file_paths: file paths of the CVS records files
    :param kapture_loadable_data: the records to load
    :param kapture_data: to the kapture object to load into
    """
    # records camera
    if kapture.RecordsCamera in kapture_loadable_data:
        records_camera_file_path = csv_file_paths[kapture.RecordsCamera]
        logger.debug(f'loading images {records_camera_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type(kapture.SENSOR_TYPE_CAMERA, kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_camera = records_camera_from_file(csv_file_paths[kapture.RecordsCamera], sensor_ids)
    # records depth
    if kapture.RecordsDepth in kapture_loadable_data:
        records_depth_file_path = csv_file_paths[kapture.RecordsDepth]
        logger.debug(f'loading depth {records_depth_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type(kapture.SENSOR_TYPE_DEPTH_CAM, kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_depth = records_depth_from_file(csv_file_paths[kapture.RecordsDepth], sensor_ids)
    # records lidar
    if kapture.RecordsLidar in kapture_loadable_data:
        records_lidar_file_path = csv_file_paths[kapture.RecordsLidar]
        logger.debug(f'loading lidar {records_lidar_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type('lidar', kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_lidar = records_lidar_from_file(records_lidar_file_path, sensor_ids)
    # records Wifi
    if kapture.RecordsWifi in kapture_loadable_data:
        records_wifi_file_path = csv_file_paths[kapture.RecordsWifi]
        logger.debug(f'loading wifi {records_wifi_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type('wifi', kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_wifi = records_wifi_from_file(records_wifi_file_path, sensor_ids)

    # records bluetooth
    if kapture.RecordsBluetooth in kapture_loadable_data:
        records_bluetooth_file_path = csv_file_paths[kapture.RecordsBluetooth]
        logger.debug(f'loading bluetooth {records_bluetooth_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type('bluetooth', kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_bluetooth = records_bluetooth_from_file(records_bluetooth_file_path, sensor_ids)

    # records GNSS
    if kapture.RecordsGnss in kapture_loadable_data:
        records_gnss_file_path = csv_file_paths[kapture.RecordsGnss]
        logger.debug(f'loading GNSS {records_gnss_file_path} ...')
        assert kapture_data.sensors is not None
        epsg_codes = {sensor_id: sensor.sensor_params[0]
                      for sensor_id, sensor in kapture_data.sensors.items()
                      if sensor.sensor_type == 'gnss'}
        if len(epsg_codes) > 0:
            kapture_data.records_gnss = records_gnss_from_file(records_gnss_file_path, set(epsg_codes.keys()))
        else:
            logger.warning('no declared GNSS sensors: all GNSS data will be ignored')

    # records Accelerometer
    if kapture.RecordsAccelerometer in kapture_loadable_data:
        records_accelerometer_file_path = csv_file_paths[kapture.RecordsAccelerometer]
        logger.debug(f'loading Accelerations {records_accelerometer_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type('accelerometer', kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_accelerometer = records_accelerometer_from_file(records_accelerometer_file_path,
                                                                             sensor_ids)
    # records Gyroscope
    if kapture.RecordsGyroscope in kapture_loadable_data:
        records_gyroscope_file_path = csv_file_paths[kapture.RecordsGyroscope]
        logger.debug(f'loading Gyroscope {records_gyroscope_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type('gyroscope', kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_gyroscope = records_gyroscope_from_file(records_gyroscope_file_path, sensor_ids)
    # records Magnetic
    if kapture.RecordsMagnetic in kapture_loadable_data:
        records_magnetic_file_path = csv_file_paths[kapture.RecordsMagnetic]
        logger.debug(f'loading Magnetic {records_magnetic_file_path} ...')
        assert kapture_data.sensors is not None
        sensor_ids = get_sensor_ids_of_type('magnetic', kapture_data.sensors)
        assert sensor_ids is not None
        kapture_data.records_magnetic = records_magnetic_from_file(records_magnetic_file_path, sensor_ids)


def list_features(kapture_type: Type[Union[kapture.Keypoints,
                                           kapture.Descriptors,
                                           kapture.GlobalFeatures]],
                  kapture_dir_path: str) -> List[str]:
    """
    list available keypoints or descriptors or global features types for a given kapture

    :param kapture_type: whether to look inside keypoints, descriptors or global features
    :param kapture_dir_path: kapture top directory path
    """
    subfolders = (
        name
        for name in os.listdir(path.join(kapture_dir_path, kapture.io.features.FEATURES_DATA_DIRNAMES[kapture_type]))
        if path.isfile(get_feature_csv_fullpath(kapture_type, name, kapture_dir_path))
    )
    return list(subfolders)


def _load_features_and_desc_and_matches(data_dir_paths: dict, kapture_dir_path: str,
                                        matches_pairs_file_path: Optional[str],
                                        kapture_loadable_data: set, kapture_data: kapture.Kapture,
                                        tar_handlers: Optional[TarCollection] = None) -> None:
    """
    Loads features, descriptors, key points and matches from disk to the kapture in memory

    :param data_dir_paths: file paths of the data records files
    :param kapture_dir_path: kapture top directory path
    :param matches_pairs_file_path: text file in the csv format; where each line is image_name1, image_name2, score
    :param kapture_loadable_data: the data to load
    :param kapture_data: to the kapture object to load into
    :param tar_handler: collection of preloaded tar archives
    """

    # features
    image_filenames = set(kapture_data.records_camera.data_list()) if kapture_data.records_camera else set()
    # keypoints
    if kapture.Keypoints in kapture_loadable_data:
        logger.debug(f'loading keypoints {data_dir_paths[kapture.Keypoints]} ...')
        assert kapture_data.records_camera is not None
        keypoints_list = list_features(kapture.Keypoints, kapture_dir_path)
        if len(keypoints_list) > 0:
            kapture_data.keypoints = {}
            for keypoints_type in keypoints_list:
                kapture_data.keypoints[keypoints_type] = keypoints_from_dir(keypoints_type,
                                                                            kapture_dir_path,
                                                                            image_filenames,
                                                                            tar_handlers)
    # descriptors
    if kapture.Descriptors in kapture_loadable_data:
        logger.debug(f'loading descriptors {data_dir_paths[kapture.Descriptors]} ...')
        assert kapture_data.records_camera is not None
        descriptors_list = list_features(kapture.Descriptors, kapture_dir_path)
        if len(descriptors_list) > 0:
            kapture_data.descriptors = {}
            for descriptors_type in descriptors_list:
                kapture_data.descriptors[descriptors_type] = descriptors_from_dir(descriptors_type,
                                                                                  kapture_dir_path,
                                                                                  image_filenames,
                                                                                  tar_handlers)
    # global_features
    if kapture.GlobalFeatures in kapture_loadable_data:
        logger.debug(f'loading global features {data_dir_paths[kapture.GlobalFeatures]} ...')
        assert kapture_data.records_camera is not None
        global_features_list = list_features(kapture.GlobalFeatures, kapture_dir_path)
        if len(global_features_list) > 0:
            kapture_data.global_features = {}
            for global_features_type in global_features_list:
                kapture_data.global_features[global_features_type] = global_features_from_dir(global_features_type,
                                                                                              kapture_dir_path,
                                                                                              image_filenames,
                                                                                              tar_handlers)
    # matches
    if kapture.Matches in kapture_loadable_data:
        logger.debug(f'loading matches {data_dir_paths[kapture.Matches]} ...')
        assert kapture_data.records_camera is not None
        matches_list = [name
                        for name in os.listdir(data_dir_paths[kapture.Matches])
                        if path.isdir(path.join(data_dir_paths[kapture.Matches], name))]
        if len(matches_list) > 0:
            kapture_data.matches = {}
            for keypoints_type in matches_list:
                kapture_data.matches[keypoints_type] = matches_from_dir(keypoints_type,
                                                                        kapture_dir_path,
                                                                        image_filenames,
                                                                        matches_pairs_file_path,
                                                                        tar_handlers)


def _load_points3d_and_observations(csv_file_paths, kapture_loadable_data, kapture_data: kapture.Kapture) -> None:
    # points3d
    if kapture.Points3d in kapture_loadable_data:
        points3d_file_path = csv_file_paths[kapture.Points3d]
        logger.debug(f'loading points 3d {points3d_file_path} ...')
        kapture_data.points3d = points3d_from_file(points3d_file_path)
    # observations
    if kapture.Observations in kapture_loadable_data:
        observations_file_path = csv_file_paths[kapture.Observations]
        logger.debug(f'loading observations {observations_file_path} ...')
        assert kapture_data.keypoints is not None
        assert kapture_data.points3d is not None
        kapture_data.observations = observations_from_file(observations_file_path, kapture_data.keypoints)


def get_all_tar_handlers(kapture_dir_path: str,  # noqa: C901: function a bit long but not too complex
                         mode: Union[str, Dict[Type, str]] = 'r',
                         skip_list: List[Type[Union[
                             kapture.Keypoints,
                             kapture.Descriptors,
                             kapture.GlobalFeatures,
                             kapture.Matches
                         ]]] = []) -> TarCollection:
    """
    Preloads all tars for the kapture data in kapture_dir_path. can be either read of append.

    :param kapture_dir_path: kapture top directory path
    :param mode: 'r' or 'a', defaults to 'r'
    :param skip_list: Input option for expert only. Skip the load of specified parts, defaults to []
    :return: collection of preloaded tar archives, don't forget to call close() when you're done with it
    """
    valid_modes = {'r', 'a'}
    if isinstance(mode, str):
        assert mode in valid_modes
    else:
        assert isinstance(mode, dict)
        for mode_t in mode.values():
            assert mode_t in valid_modes

    data_dir_paths = {dtype: path.join(kapture_dir_path, dir_name)
                      for dtype, dir_name in kapture.io.features.FEATURES_DATA_DIRNAMES.items()}
    kapture_loadable_data = {
        kapture_type
        for kapture_type in KAPTURE_TARABLE_TYPES
        if kapture_type not in skip_list and path.exists(data_dir_paths[kapture_type])
    }
    tar_collection = TarCollection()

    # keypoints
    if kapture.Keypoints in kapture_loadable_data:
        logger.debug(f'opening keypoints tars {data_dir_paths[kapture.Keypoints]} ...')
        keypoints_list = list_features(kapture.Keypoints, kapture_dir_path)
        opened_tar_count = 0
        if len(keypoints_list) > 0:
            if isinstance(mode, str):
                mode_t = mode
            elif kapture.Keypoints in mode:
                mode_t = mode[kapture.Keypoints]
            else:
                mode_t = 'r'
            for keypoints_type in keypoints_list:
                tarfile_path = get_feature_tar_fullpath(kapture.Keypoints, keypoints_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.keypoints[keypoints_type] = TarHandler(tarfile_path, mode_t)
                    opened_tar_count += 1
        logger.debug(f'opened {opened_tar_count} keypoints tars')
    # descriptors
    if kapture.Descriptors in kapture_loadable_data:
        logger.debug(f'opening descriptors tars {data_dir_paths[kapture.Descriptors]} ...')
        descriptors_list = list_features(kapture.Descriptors, kapture_dir_path)
        opened_tar_count = 0
        if len(descriptors_list) > 0:
            if isinstance(mode, str):
                mode_t = mode
            elif kapture.Descriptors in mode:
                mode_t = mode[kapture.Descriptors]
            else:
                mode_t = 'r'
            for descriptors_type in descriptors_list:
                tarfile_path = get_feature_tar_fullpath(kapture.Descriptors, descriptors_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.descriptors[descriptors_type] = TarHandler(tarfile_path, mode_t)
                    opened_tar_count += 1
        logger.debug(f'opened {opened_tar_count} descriptors tars')
    # global_features
    if kapture.GlobalFeatures in kapture_loadable_data:
        logger.debug(f'opening global_features tars {data_dir_paths[kapture.GlobalFeatures]} ...')
        global_features_list = list_features(kapture.GlobalFeatures, kapture_dir_path)
        opened_tar_count = 0
        if len(global_features_list) > 0:
            if isinstance(mode, str):
                mode_t = mode
            elif kapture.GlobalFeatures in mode:
                mode_t = mode[kapture.GlobalFeatures]
            else:
                mode_t = 'r'
            for global_features_type in global_features_list:
                tarfile_path = get_feature_tar_fullpath(kapture.GlobalFeatures, global_features_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.global_features[global_features_type] = TarHandler(tarfile_path, mode_t)
                    opened_tar_count += 1
        logger.debug(f'opened {opened_tar_count} global_features tars')
    # matches
    if kapture.Matches in kapture_loadable_data:
        logger.debug(f'opening matches tars {data_dir_paths[kapture.Matches]} ...')
        keypoints_list = [name
                          for name in os.listdir(data_dir_paths[kapture.Matches])
                          if os.path.isdir(os.path.join(data_dir_paths[kapture.Matches], name))]
        opened_tar_count = 0
        if len(keypoints_list) > 0:
            if isinstance(mode, str):
                mode_t = mode
            elif kapture.Matches in mode:
                mode_t = mode[kapture.Matches]
            else:
                mode_t = 'r'
            for keypoints_type in keypoints_list:
                tarfile_path = get_feature_tar_fullpath(kapture.Matches, keypoints_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.matches[keypoints_type] = TarHandler(tarfile_path, mode_t)
                    opened_tar_count += 1
        logger.debug(f'opened {opened_tar_count} matches tars')
    return tar_collection
