# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
All reading and writing operations of kapture objects in CSV like files
"""

import os
import os.path as path
import re
from typing import Any, List, Optional, Set, Type, Union
from collections import namedtuple
import numpy as np

import kapture
import kapture.io.features

logger = kapture.logger

# file names conventions
CSV_FILENAMES = {
    kapture.Sensors: path.join('sensors', 'sensors.txt'),
    kapture.Trajectories: path.join('sensors', 'trajectories.txt'),
    kapture.Rigs: path.join('sensors', 'rigs.txt'),
    kapture.RecordsCamera: path.join('sensors', 'records_camera.txt'),
    kapture.RecordsLidar: path.join('sensors', 'records_lidar.txt'),
    kapture.RecordsWifi: path.join('sensors', 'records_wifi.txt'),
    kapture.RecordsGnss: path.join('sensors', 'records_gnss.txt'),
    kapture.Points3d: path.join('reconstruction', 'points3d.txt'),
    kapture.Keypoints: path.join('reconstruction', 'keypoints', 'keypoints.txt'),
    kapture.Descriptors: path.join('reconstruction', 'descriptors', 'descriptors.txt'),
    kapture.GlobalFeatures: path.join('reconstruction', 'global_features', 'global_features.txt'),
    kapture.Observations: path.join('reconstruction', 'observations.txt'),
}


def get_csv_fullpath(kapture_type: Any, kapture_dirpath: str = '') -> str:
    """
    Returns the full path to csv kapture file for a given datastructure and root directory.
    This path is the concatenation of the kapture root path and subpath into kapture into data structure.

    :param kapture_type: type of kapture data (kapture.RecordsCamera, kapture.Trajectories, ...)
    :param kapture_dirpath: root kapture path
    :return: full path of csv file for that type of data
    """
    filename = CSV_FILENAMES[kapture_type]
    return path.join(kapture_dirpath, filename)


PADDINGS = {
    'timestamp': [8],
    'device_id': [3],
    'pose': [4, 4, 4, 4, 4, 4, 4],
}

KAPTURE_FORMAT_1 = "# kapture format: 1.0"
KAPTURE_FORMAT_PARSING_RE = '# kapture format\\:\\s*(?P<version>\\d+\\.\\d+)'


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


def current_format_version() -> str:
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


def float_safe(representation) -> Union[float, None]:
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


def float_array_or_none(representation_list) -> Union[List[float], None]:
    """
    Safe cast of list of float representations
    https://stackoverflow.com/questions/6330071/safe-casting-in-python

    :param representation_list: list of values to convert
    :return: an array of floats or None if a single one is invalid
    """
    array = [float_safe(v) for v in representation_list]
    return array if not any(v is None for v in array) else None


def table_to_file(file, table, header=None, padding=None) -> None:
    """
    Writes the given table (list of list) into a file.
            The file must be previously open as write mode.
            If table is a generator, must be valid at runtime.

    :param file: file id opened in write mode (with open(filepath, 'w') as file:)
    :param table: an iterable of iterable
    :param header: row added at the beginning of the file (+\n)
    :param padding: the padding of each column as a list of int of same size of the rows of the table.
    :return:
    """
    if header:
        file.write(KAPTURE_FORMAT_1 + '\n')
        file.write(header + '\n')
    for row in table:
        if padding:
            row = [str(v).rjust(padding[i]) for i, v in enumerate(row)]
        file.write(', '.join(f'{v}' for v in row) + '\n')


def table_from_file(file):
    """
    Returns an iterable of iterable (generator) on the opened file.
        Be aware that the returned generator is valid as long as file is valid.

    :param file: file id opened in read mode (with open(filepath, 'r') as file:)
    :return: an iterable of iterable on kapture objects values
    """
    table = file.readlines()
    # remove comment lines
    table = (l1 for l1 in table if not l1.startswith('#'))
    # remove empty lines
    table = (l2 for l2 in table if l2.strip())
    # trim trailing EOL
    table = (l3.rstrip("\n\r") for l3 in table)
    # split comma (and trim afterwards spaces)
    table = (re.split(r'\s*,\s*', l4) for l4 in table)
    return table


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


def rigs_from_file(filepath: str, sensor_ids: Set[str]) -> kapture.Rigs:
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
            if rig_id in sensor_ids:
                raise ValueError(f'collision between a sensor ID and rig ID ({rig_id})')
            if sensor_id not in sensor_ids:
                # just ignore
                continue
            rotation = float_array_or_none([qw, qx, qy, qz])
            translation = float_array_or_none([tx, ty, tz])
            pose = kapture.PoseTransform(rotation, translation)
            # rigs.setdefault(str(rig_id), kapture.Rig())[sensor_id] = pose
            rigs[str(rig_id), sensor_id] = pose
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
    header = '# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz'
    padding = PADDINGS['timestamp'] + PADDINGS['device_id'] + PADDINGS['pose']
    table = (
        [timestamp, sensor_id] + pose_to_list(trajectories[(timestamp, sensor_id)])
        for timestamp, sensor_id in sorted(trajectories.key_pairs())
    )

    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header, padding=padding)


def trajectories_from_file(filepath: str, device_ids: Optional[Set[str]] = None) -> kapture.Trajectories:
    """
    Reads trajectories from CSV file.

    :param filepath: input file path
    :param device_ids: input set of valid device ids (rig or sensor).
                        If the trajectories contains unknown devices, they will be ignored.
                        If no device_ids given, everything is loaded.
    :return: trajectories
    """
    trajectories = kapture.Trajectories()
    with open(filepath) as file:
        table = table_from_file(file)
        # timestamp, device_id, qw, qx, qy, qz, tx, ty, tz
        for timestamp, device_id, qw, qx, qy, qz, tx, ty, tz in table:
            if device_ids is not None and device_id not in device_ids:
                # just ignore
                continue
            rotation = float_array_or_none([qw, qx, qy, qz])
            trans = float_array_or_none([tx, ty, tz])
            pose = kapture.PoseTransform(rotation, trans)
            trajectories[(int(timestamp), str(device_id))] = pose
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
    header = '# timestamp, device_id, image_path'
    table = (
        [timestamp, sensor_id] + [records_camera[(timestamp, sensor_id)]]
        for timestamp, sensor_id in sorted(records_camera.key_pairs())
    )
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header)


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
    with open(filepath) as file:
        table = table_from_file(file)
        # timestamp, device_id, image_path
        for timestamp, device_id, image_path in table:
            if camera_ids is not None and device_id not in camera_ids:
                # just ignore
                continue
            records_camera[(int(timestamp), str(device_id))] = image_path
    return records_camera


########################################################################################################################
# Records Lidar ########################################################################################################
def records_lidar_to_file(filepath: str, records_lidar: kapture.RecordsLidar) -> None:
    """
    Writes records_lidar to CSV file.

    :param filepath:
    :param records_lidar:
    """
    assert (isinstance(records_lidar, kapture.RecordsLidar))
    header = '# timestamp, device_id, point_cloud_path'
    table = (
        [timestamp, sensor_id] + [records_lidar[(timestamp, sensor_id)]]
        for timestamp, sensor_id in sorted(records_lidar.key_pairs())
    )
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header)


def records_lidar_from_file(
        filepath: str,
        lidar_ids: Optional[Set[str]] = None
) -> kapture.RecordsLidar:
    """
    Reads records_lidar from CSV file.

    :param filepath: input file path
    :param lidar_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                            If omitted, then it loads all devices.
    :return: Lidar records
    """
    records_lidar = kapture.RecordsLidar()
    with open(filepath) as file:
        table = table_from_file(file)
        # timestamp, device_id, point_cloud_path
        for timestamp, device_id, point_cloud_path in table:
            if lidar_ids is not None and device_id not in lidar_ids:
                # just ignore
                continue
            records_lidar[(int(timestamp), str(device_id))] = point_cloud_path
    return records_lidar


########################################################################################################################
# Records Wifi #########################################################################################################
def records_wifi_to_file(filepath: str, records_wifi: kapture.RecordsWifi) -> None:
    """
    Writes records_wifi to file

    :param filepath:
    :param records_wifi:
    """
    assert (isinstance(records_wifi, kapture.RecordsWifi))
    header = ('# timestamp (ScanEndTime), device_id, '
              'BSSID, RSSI, FREQ, SCANTIME, VISIBLENAME')
    table = []
    for timestamp, sensor_id in sorted(records_wifi.key_pairs()):
        for bssid in records_wifi[timestamp, sensor_id]:
            table.append([timestamp, sensor_id, bssid] + records_wifi[(timestamp, sensor_id)][bssid].as_list())
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header)


def records_wifi_from_file(
        filepath: str,
        wifi_ids: Optional[Set[str]] = None
) -> kapture.RecordsWifi:
    """
    Reads RecordsWifi from CSV file.

    :param filepath: input file path
    :param wifi_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                            If omitted, then it loads all devices.
    :return: Wifi records
    """
    records_wifi = kapture.RecordsWifi()
    with open(filepath) as file:
        table = table_from_file(file)
        records_wifi_current = {}
        cur_timestamp = -1
        # timestamp (ScanEndTime), device_id, BSSID, RSSI, FREQ, SCANTIME, VISIBLENAME
        for timestamp, device_id, bssid, *wifi_params in table:
            if wifi_ids is not None and device_id not in wifi_ids:
                # just ignore
                continue
            if timestamp != cur_timestamp:
                if records_wifi_current:
                    records_wifi[(int(cur_timestamp), str(device_id))] = records_wifi_current
                records_wifi_current = {}
                cur_timestamp = timestamp
            records_wifi_current[bssid] = kapture.RecordWifi(*wifi_params)
        # Don't forget last line
        if records_wifi_current:
            records_wifi[(int(cur_timestamp), str(device_id))] = records_wifi_current

    return records_wifi


########################################################################################################################
# Records GNSS #########################################################################################################
def records_gnss_to_file(
        filepath: str,
        records_gnss: kapture.RecordsGnss
) -> None:
    """
    Writes records_gnss to file

    :param filepath:
    :param records_gnss:
    """
    assert isinstance(records_gnss, kapture.RecordsGnss)
    header = '# timestamp, device_id, x, y, z, utc, dop'
    table = []
    for timestamp, sensor_id, gnss_record in kapture.flatten(records_gnss, is_sorted=True):
        table.append([timestamp, sensor_id] + gnss_record.as_list())
    with open(filepath, 'w') as file:
        table_to_file(file, table, header=header)


def records_gnss_from_file(
        filepath: str,
        gnss_ids: Optional[Set[str]] = None
) -> kapture.RecordsGnss:
    """
    Reads RecordsGnss from CSV file.

    :param filepath: input file path
    :param gnss_ids: input set of valid device ids. Any record of other than the given ones will be ignored.
                     If omitted, then it loads all devices.
    :return: GNSS records
    """
    records_gnss = kapture.RecordsGnss()
    with open(filepath) as file:
        table = table_from_file(file)
        # timestamp, device_id, x, y, z, utc, dop
        for timestamp, device_id, x, y, z, utc, dop in table:
            timestamp = int(timestamp)
            device_id = str(device_id)
            if gnss_ids is not None and device_id not in gnss_ids:
                # just ignore
                continue
            records_gnss[timestamp, device_id] = kapture.RecordGnss(x, y, z, utc, dop)

    return records_gnss


########################################################################################################################
# features #############################################################################################################
ImageFeatureConfig = namedtuple('ImageFeatureConfig', ['name', 'dtype', 'dsize'])


# config file ##########################################################################################################
def image_features_config_to_file(config_filepath: str, config: ImageFeatureConfig) -> None:
    """
    Writes feature config files, ie. name, data type, and data size.

    :param config_filepath: input path to config file to write.
    :param config: input config to be written.
    """
    os.makedirs(path.dirname(config_filepath), exist_ok=True)
    header = "# name, dtype, dsize"
    dtype = str(config.dtype) if isinstance(config.dtype, np.dtype) else config.dtype.__name__
    line = [config.name, dtype, str(config.dsize)]
    with open(config_filepath, 'wt') as file:
        table_to_file(file, [line], header=header)


def image_features_config_from_file(config_filepath: str) -> ImageFeatureConfig:
    """
    Reads feature config files, ie. name, data type, and data size.

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
    from numpy import float, float32, float64, int32, uint8
    if isinstance(type(eval(dtype)), type):
        dtype = eval(dtype)
    else:
        raise ValueError('Expect data type ')
    return ImageFeatureConfig(name, dtype, dsize)


# files #########################################################################################################
def array_to_file(
        config_filepath: str,
        image_features: Union[kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures]
) -> None:
    """
    Writes ImageFeatures config file only.

    :param config_filepath: eg. /path/to/keypoints/keypoints.txt
    :param image_features: input feature date.
    """
    config = ImageFeatureConfig(image_features.type_name, image_features.dtype, image_features.dsize)
    image_features_config_to_file(config_filepath, config)


def image_features_from_dir(
        kapture_type: Type[Union[kapture.Keypoints,
                                 kapture.Descriptors,
                                 kapture.GlobalFeatures]],
        kapture_dirpath: str,
        image_filenames: Optional[Set[str]]
) -> kapture.ImageFeatures:
    """
    Reads and builds ImageFeatures from images_filenames if given, or directly from actual files in  root_dirpath.

    :param kapture_type: kapture class type.
    :param kapture_dirpath: input path to kapture root directory.
    :param image_filenames: None or set of image relative paths.
    :return: Features
    """

    # make config_filepath from data_dirpath
    config_filepath = get_csv_fullpath(kapture_type, kapture_dirpath)
    config = image_features_config_from_file(config_filepath)
    if image_filenames is None:
        # images_path is empty, so populates all feature files
        image_filenames_generator = kapture.io.features.image_ids_from_feature_dirpath(kapture_type, kapture_dirpath)
    else:
        # filter only existing files
        image_filenames_generator = (
            image_name
            for image_name in image_filenames
            if path.exists(kapture.io.features.get_features_fullpath(kapture_type, kapture_dirpath, image_name)))

    image_filenames = set(image_filenames_generator)
    return kapture_type(config.name, config.dtype, config.dsize, image_filenames)


########################################################################################################################
# keypoints ############################################################################################################
def keypoints_to_file(config_filepath: str, keypoints: kapture.Keypoints) -> None:
    """
    Writes keypoints to CSV file.

    :param config_filepath:
    :param keypoints:
    """
    return array_to_file(config_filepath=config_filepath,
                         image_features=keypoints)


def keypoints_from_dir(kapture_dirpath: str, images_paths: Optional[Set[str]]) -> kapture.Keypoints:
    """
    Reads and builds keypoints from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param kapture_dirpath: root path of kapture
    :param images_paths: optional list of image file names
    :return: Keypoints
    """
    return image_features_from_dir(kapture_type=kapture.Keypoints,
                                   kapture_dirpath=kapture_dirpath,
                                   image_filenames=images_paths)


########################################################################################################################
# descriptors ##########################################################################################################
def descriptors_to_file(config_filepath: str, descriptors: kapture.Descriptors) -> None:
    """
    Writes descriptors to CSV file.

    :param config_filepath:
    :param descriptors:
    """
    return array_to_file(config_filepath=config_filepath,
                         image_features=descriptors)


def descriptors_from_dir(kapture_dirpath: str, images_paths: Set[str]) -> kapture.Descriptors:
    """
    Reads and builds descriptors from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param kapture_dirpath: root path of kapture
    :param images_paths: optional list of image file names
    :return: Descriptors
    """
    return image_features_from_dir(kapture_type=kapture.Descriptors,
                                   kapture_dirpath=kapture_dirpath,
                                   image_filenames=images_paths)


########################################################################################################################
# global_features ######################################################################################################
def global_features_to_file(config_filepath: str, global_features: kapture.GlobalFeatures) -> None:
    """
    Writes global features to CSV file.

    :param config_filepath:
    :param global_features:
    """
    return array_to_file(config_filepath=config_filepath,
                         image_features=global_features)


def global_features_from_dir(kapture_dirpath: str, images_paths: Set[str]) -> kapture.GlobalFeatures:
    """
    Reads and builds Global features from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param kapture_dirpath: root path of kapture
    :param images_paths: optional list of image file names
    :return: Global features
    """

    return image_features_from_dir(kapture_type=kapture.GlobalFeatures,
                                   kapture_dirpath=kapture_dirpath,
                                   image_filenames=images_paths)


########################################################################################################################
# matches ##############################################################################################################
def matches_from_dir(kapture_dirpath: str,
                     image_filenames: Optional[Set[str]] = None,
                     matches_pairsfile_path: Optional[str] = None) -> kapture.Matches:
    """
    Reads and builds Matches from images_filenames if given, or directly from actual files in kapture_dirpath.

    :param kapture_dirpath: root path of kapture
    :param image_filenames: optional list of image file names
    :param matches_pairsfile_path: text file in the csv format; where each line is image_name1, image_name2, score
    :return: Matches
    """

    if matches_pairsfile_path is None:
        # populate files from disk
        match_pairs_generator = kapture.io.features.matching_pairs_from_dirpath(kapture_dirpath)
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
    header = 'X, Y, Z, R, G, B'
    np.savetxt(filepath, points3d.as_array(), delimiter=',', header=header)


def points3d_from_file(filepath: str) -> kapture.Points3d:
    """
    Reads 3d points from CSV file.

    :param filepath: path to CSV file
    :return: the 3d points
    """

    data = np.loadtxt(filepath, dtype=np.float, delimiter=',', comments='#')
    data = data.reshape((-1, 6))  # make sure of the shape, even if single line file.
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
    assert path.basename(observations_filepath) == 'observations.txt'
    assert isinstance(observations, kapture.Observations)
    header = '# point3d_id, [image_path, feature_id]*'
    table = (
        [str(point3d_idx)] + [str(k) for pair in observations[point3d_idx] for k in pair]
        for point3d_idx in sorted(observations.keys())
    )
    os.makedirs(path.dirname(observations_filepath), exist_ok=True)
    with open(observations_filepath, 'w') as file:
        table_to_file(file, table, header=header)


def observations_from_file(
        observations_filepath: str,
        images_paths_with_keypoints: Optional[Set[str]] = None
) -> kapture.Observations:
    """
    Reads observations from CSV file.

    :param observations_filepath: path to CSV file to read.
    :param images_paths_with_keypoints: input set of image names (ids) that have keypoints.
                                        If given, used to filter out irrelevant observations.
                                        You can get from set(kapture.keypoints)
    :return: observations
    """
    assert path.basename(observations_filepath) == 'observations.txt'
    assert images_paths_with_keypoints is None or \
           (isinstance(images_paths_with_keypoints, set) and len(images_paths_with_keypoints) > 0)
    observations = kapture.Observations()
    with open(observations_filepath) as file:
        table = table_from_file(file)
        # point3d_id, [image_path, feature_id]*
        for points3d_id_str, *pairs in table:
            points3d_id = int(points3d_id_str)
            if len(pairs) > 1:
                image_paths = pairs[0::2]
                keypoints_ids = pairs[1::2]
                for image_path, keypoint_id in zip(image_paths, keypoints_ids):
                    if images_paths_with_keypoints is not None and image_path not in images_paths_with_keypoints:
                        # image_path does not exist in kapture (perhaps it was removed), ignore it
                        continue
                    observations.add(points3d_id, image_path, int(keypoint_id))
    return observations


########################################################################################################################
# Kapture Write ########################################################################################################
KAPTURE_ATTRIBUTE_WRITERS = {
    kapture.Sensors: sensors_to_file,
    kapture.Rigs: rigs_to_file,
    kapture.Trajectories: trajectories_to_file,
    kapture.RecordsCamera: records_camera_to_file,
    kapture.RecordsLidar: records_lidar_to_file,
    kapture.RecordsWifi: records_wifi_to_file,
    kapture.RecordsGnss: records_gnss_to_file,
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
    kapture.RecordsLidar: 'records_lidar',
    kapture.RecordsWifi: 'records_wifi',
    kapture.RecordsGnss: 'records_gnss',
    kapture.Keypoints: 'keypoints',
    kapture.Descriptors: 'descriptors',
    kapture.GlobalFeatures: 'global_features',
    kapture.Points3d: 'points3d',
    kapture.Observations: 'observations',
}


def kapture_to_dir(dirpath: str, kapture_data: kapture.Kapture) -> None:
    """
    Saves kapture data to given directory.

    :param dirpath: input directory root path
    :param kapture_data: input kapture data
    """
    kapture_subtype_to_filepaths = {kapture_class: path.join(dirpath, filename)
                                    for kapture_class, filename in CSV_FILENAMES.items()}
    # save each member of kapture data
    for kapture_class, kapture_member_name in KAPTURE_ATTRIBUTE_NAMES.items():
        part_data = kapture_data.__getattribute__(kapture_member_name)
        if part_data is not None:
            # save it
            logger.debug(f'saving {kapture_member_name} ...')
            write_function = KAPTURE_ATTRIBUTE_WRITERS[kapture_class]
            write_function(kapture_subtype_to_filepaths[kapture_class], part_data)


# Kapture Read #########################################################################################################
# list all data members of kapture.
KAPTURE_LOADABLE_TYPES = {
    kapture.Sensors,
    kapture.Rigs,
    kapture.Trajectories,
    kapture.RecordsCamera,
    kapture.RecordsLidar,
    kapture.RecordsWifi,
    kapture.RecordsGnss,
    kapture.Keypoints,
    kapture.Descriptors,
    kapture.GlobalFeatures,
    kapture.Matches,
    kapture.Points3d,
    kapture.Observations,
}


def kapture_from_dir(
        kapture_dirpath: str,
        matches_pairsfile_path: Optional[str] = None,
        skip_list: List[Type[Union[kapture.Rigs,
                                   kapture.Trajectories,
                                   kapture.RecordsCamera,
                                   kapture.RecordsLidar,
                                   kapture.RecordsWifi,
                                   kapture.RecordsGnss,
                                   kapture.Keypoints,
                                   kapture.Descriptors,
                                   kapture.GlobalFeatures,
                                   kapture.Matches,
                                   kapture.Points3d,
                                   kapture.Observations,

                                   ]]] = []
) -> kapture.Kapture:
    """
    Reads and return kapture data from directory.

    :param kapture_dirpath: kapture directory root path
    :param matches_pairsfile_path: text file in the csv format; where each line is image_name1, image_name2, score
    :param skip_list: Input option for expert only. Skip the load of specified parts.
    :return: kapture data read
    """
    if not path.isdir(kapture_dirpath):
        raise FileNotFoundError(f'No kapture directory {kapture_dirpath}')
    csv_filepaths = {dtype: path.join(kapture_dirpath, filename)
                     for dtype, filename in CSV_FILENAMES.items()}
    data_dirpaths = {dtype: path.join(kapture_dirpath, dir_name)
                     for dtype, dir_name in kapture.io.features.FEATURES_DATA_DIRNAMES.items()}

    # keep only those in load_only and that exists
    kapture_data_paths = {**data_dirpaths, **csv_filepaths}  # make sure files take precedence over dirs
    kapture_loadable_data = {
        kapture_type
        for kapture_type in KAPTURE_LOADABLE_TYPES
        if kapture_type not in skip_list and path.exists(kapture_data_paths[kapture_type])
    }

    kapture_data = kapture.Kapture()
    # sensors
    sensor_ids = None
    sensors_file_path = csv_filepaths[kapture.Sensors]
    if sensors_file_path:
        logger.debug(f'loading sensors {sensors_file_path} ...')
        kapture_data.__version__ = get_version_from_csv_file(sensors_file_path)
        kapture_data.sensors = sensors_from_file(sensors_file_path)
        sensor_ids = set(kapture_data.sensors.keys()) if kapture_data.sensors is not None else set()

    if sensor_ids is None:
        # no need to continue, everything else depends on sensors
        raise FileNotFoundError(f'File {sensors_file_path} is missing or empty in {kapture_dirpath}')

    # rigs
    if kapture.Rigs in kapture_loadable_data:
        rigs_file_path = csv_filepaths[kapture.Rigs]
        logger.debug(f'loading rigs {rigs_file_path} ...')
        assert sensor_ids is not None
        kapture_data.rigs = rigs_from_file(rigs_file_path, sensor_ids)
        # update sensor_ids with rig_id
        sensor_ids.update(kapture_data.rigs.keys())

    # trajectories
    if kapture.Trajectories in kapture_loadable_data:
        trajectories_file_path = csv_filepaths[kapture.Trajectories]
        logger.debug(f'loading trajectories {trajectories_file_path} ...')
        assert sensor_ids is not None
        kapture_data.trajectories = trajectories_from_file(trajectories_file_path, sensor_ids)

    # records camera
    if kapture.RecordsCamera in kapture_loadable_data:
        records_camera_file_path = csv_filepaths[kapture.RecordsCamera]
        logger.debug(f'loading images {records_camera_file_path} ...')
        assert kapture_data.sensors is not None
        camera_sensor_ids = set([sensor_id
                                 for sensor_id in kapture_data.sensors.keys()
                                 if kapture_data.sensors[sensor_id].sensor_type == 'camera'])
        kapture_data.records_camera = records_camera_from_file(csv_filepaths[kapture.RecordsCamera], camera_sensor_ids)

    # records lidar
    if kapture.RecordsLidar in kapture_loadable_data:
        records_lidar_file_path = csv_filepaths[kapture.RecordsLidar]
        logger.debug(f'loading lidar {records_lidar_file_path} ...')
        assert kapture_data.sensors is not None
        lidar_sensor_ids = set([sensor_id
                                for sensor_id in kapture_data.sensors.keys()
                                if kapture_data.sensors[sensor_id].sensor_type == 'lidar'])
        assert lidar_sensor_ids is not None
        kapture_data.records_lidar = records_lidar_from_file(records_lidar_file_path, lidar_sensor_ids)

    # records Wifi
    if kapture.RecordsWifi in kapture_loadable_data:
        records_wifi_file_path = csv_filepaths[kapture.RecordsWifi]
        logger.debug(f'loading wifi {records_wifi_file_path} ...')
        assert kapture_data.sensors is not None
        wifi_sensor_ids = set([sensor_id
                               for sensor_id in kapture_data.sensors.keys()
                               if kapture_data.sensors[sensor_id].sensor_type == 'wifi'])
        assert wifi_sensor_ids is not None
        kapture_data.records_wifi = records_wifi_from_file(records_wifi_file_path, wifi_sensor_ids)

    # records GNSS
    if kapture.RecordsGnss in kapture_loadable_data:
        records_gnss_file_path = csv_filepaths[kapture.RecordsGnss]
        logger.debug(f'loading GNSS {records_gnss_file_path} ...')
        assert kapture_data.sensors is not None
        epsg_codes = {sensor_id: sensor.sensor_params[0]
                      for sensor_id, sensor in kapture_data.sensors.items()
                      if sensor.sensor_type == 'gnss'}
        if len(epsg_codes) > 0:
            kapture_data.records_gnss = records_gnss_from_file(records_gnss_file_path, epsg_codes)
        else:
            logger.warning('no declared GNSS sensors: all GNSS data will be ignored')

    # features
    image_filenames = set(image_name
                          for _, _, image_name in
                          kapture.flatten(kapture_data.records_camera)) \
        if kapture_data.records_camera is not None else set()

    # keypoints
    if kapture.Keypoints in kapture_loadable_data:
        logger.debug(f'loading keypoints {data_dirpaths[kapture.Keypoints]} ...')
        assert kapture_data.records_camera is not None
        kapture_data.keypoints = keypoints_from_dir(kapture_dirpath, image_filenames)

    # descriptors
    if kapture.Descriptors in kapture_loadable_data:
        logger.debug(f'loading descriptors {data_dirpaths[kapture.Descriptors]} ...')
        assert kapture_data.records_camera is not None
        kapture_data.descriptors = descriptors_from_dir(kapture_dirpath, image_filenames)

    # global_features
    if kapture.GlobalFeatures in kapture_loadable_data:
        logger.debug(f'loading global features {data_dirpaths[kapture.GlobalFeatures]} ...')
        assert kapture_data.records_camera is not None
        kapture_data.global_features = global_features_from_dir(kapture_dirpath, image_filenames)

    # matches
    if kapture.Matches in kapture_loadable_data:
        logger.debug(f'loading matches {data_dirpaths[kapture.Matches]} ...')
        assert kapture_data.records_camera is not None
        kapture_data.matches = matches_from_dir(kapture_dirpath, image_filenames, matches_pairsfile_path)

    # points3d
    if kapture.Points3d in kapture_loadable_data:
        points3d_file_path = csv_filepaths[kapture.Points3d]
        logger.debug(f'loading points 3d {points3d_file_path} ...')
        kapture_data.points3d = points3d_from_file(points3d_file_path)

    # observations
    if kapture.Observations in kapture_loadable_data:
        observations_file_path = csv_filepaths[kapture.Observations]
        logger.debug(f'loading observations {observations_file_path} ...')
        assert kapture_data.keypoints is not None
        assert kapture_data.points3d is not None
        kapture_data.observations = observations_from_file(observations_file_path, kapture_data.keypoints)

    return kapture_data
