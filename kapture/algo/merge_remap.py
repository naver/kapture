# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Merge kapture objects with remapping of identifiers.
"""

from typing import List, Optional, Type, Dict

import kapture
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.utils.Collections import get_new_if_not_empty

from .merge_reconstruction import merge_keypoints, merge_descriptors, merge_global_features, merge_matches
from .merge_reconstruction import merge_points3d_and_observations, merge_points3d
from .merge_records_data import merge_records_data


def get_sensors_mapping(sensors: kapture.Sensors, offset: int = 0) -> Dict[str, str]:
    """
    Creates list of sensor names,identifiers

    :param sensors: list of sensor definitions
    :param offset: optional offset for the identifier numbers
    :return: mapping of sensor names to identifiers
    """
    return {k: f'sensor{v}' for k, v in zip(sensors.keys(), range(offset, offset + len(sensors)))}


def get_rigs_mapping(rigs: kapture.Rigs, offset: int = 0) -> Dict[str, str]:
    """
    Creates list of rig names,identifiers

    :param rigs: list of rig definitions
    :param offset: optional offset for the identifier numbers
    :return: mapping of rig names to identifiers
    """
    return {k: f'rig{v}' for k, v in zip(rigs.keys(), range(offset, offset + len(rigs)))}


def merge_table_key1(
        table_list,
        sensor_mappings: List[Dict[str, str]],
        table_constructor,
):
    """
    Merge several table with 1 key (Only device_id) into one.
    If multiple entry for a key keep only the first one.

    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :param table_list: list of table to merge.
    :param table_constructor: the class type of table.
    :return table_merged

    """
    assert len(table_list) > 0
    assert len(table_list) == len(sensor_mappings)
    table_list = [table for table in table_list if table is not None]
    if not all(isinstance(table, table_constructor) for table in table_list):
        raise TypeError('unexpected type.')
    table_merged = table_constructor()
    for table, sensor_mapping in zip(table_list, sensor_mappings):
        for sensor_id, entry in kapture.flatten(table):
            new_sensor_id = sensor_mapping[sensor_id]
            table_merged[new_sensor_id] = entry
    return table_merged


def merge_table_key2(
        table_list,
        sensor_mappings: List[Dict[str, str]],
        table_constructor,
):
    """
    Merge several table with 2 keys (eg. timestamps, device_id)  into one.
    If multiple entry for a key keep only the first one.

    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :param table_list: list of table to merge.
    :param table_constructor: the class type of table.
    :return table_merged

    """
    assert len(table_list) > 0
    assert len(table_list) == len(sensor_mappings)
    table_list = [table for table in table_list if table is not None]
    if not all(isinstance(table, table_constructor) for table in table_list):
        raise TypeError('unexpected type.')
    table_merged = table_constructor()
    for table, sensor_mapping in zip(table_list, sensor_mappings):
        for key1, sensor_id, entry in kapture.flatten(table):
            new_sensor_id = sensor_mapping[sensor_id]
            table_merged[key1, new_sensor_id] = entry
    return table_merged


def merge_table_key3(
        table_list,
        sensor_mappings: List[Dict[str, str]],
        table_constructor,
        subdict_constructor=dict,
):
    """
    Merge several table with 2 keys (eg. timestamps, device_id)  into one.
    If multiple entry for a key keep only the first one.

    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :param table_list: list of table to merge.
    :param table_constructor: the class type of table.
    :param subdict_constructor: used to create a new Dict type
    :return table_merged

    """
    assert len(table_list) > 0
    assert len(table_list) == len(sensor_mappings)
    table_list = [table for table in table_list if table is not None]
    if not all(isinstance(table, table_constructor) for table in table_list):
        raise TypeError('unexpected type.')
    table_merged = table_constructor()
    for table, sensor_mapping in zip(table_list, sensor_mappings):
        for key1, sensor_id, key3, entry in kapture.flatten(table):
            new_sensor_id = sensor_mapping[sensor_id]
            if (key1, new_sensor_id) not in table_merged:
                # if timestamp, sensor_id not there yet, create an instance of dict record
                table_merged[key1, new_sensor_id] = subdict_constructor()
            table_merged[key1, new_sensor_id].setdefault(key3, entry)
    return table_merged


def merge_sensors(
        sensors_list: List[Optional[kapture.Sensors]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.Sensors:
    """
    Merge several sensors list into one list with new identifiers.

    :param sensors_list: list of sensors definitions to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged sensors definitions
    """
    return merge_table_key1(
        table_list=sensors_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.Sensors
    )


def merge_rigs(
        rigs_list: List[Optional[kapture.Rigs]],
        rig_mappings: List[Dict[str, str]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.Rigs:
    """
    Merge several rigs list into one list with new identifiers for the rigs and the sensors.

    :param rigs_list: list of rigs definitions to merge
    :param rig_mappings: mapping of the rigs identifiers to their new identifiers
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged rigs definitions
    """
    assert len(rigs_list) > 0
    assert len(rigs_list) == len(rig_mappings)
    assert len(rigs_list) == len(sensor_mappings)

    merged_rigs = kapture.Rigs()
    for rigs, rig_mapping, sensor_mapping in zip(rigs_list, rig_mappings, sensor_mappings):
        if rigs is None:
            continue
        for rig_id, sensor_id in rigs.key_pairs():
            new_rig_id = rig_mapping[rig_id]
            new_sensor_id = sensor_mapping[sensor_id]
            merged_rigs[(new_rig_id, new_sensor_id)] = rigs[(rig_id, sensor_id)]
    return merged_rigs


def merge_trajectories(
        trajectories_list: List[Optional[kapture.Trajectories]],
        rig_mappings: List[Dict[str, str]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.Trajectories:
    """
    Merge several trajectories list into one list with new identifiers for the rigs and the sensors.

    :param trajectories_list: list of trajectories to merge
    :param rig_mappings: mapping of the rigs identifiers to their new identifiers
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged trajectories
    """
    assert len(trajectories_list) > 0
    assert len(trajectories_list) == len(rig_mappings)
    assert len(trajectories_list) == len(sensor_mappings)

    merged_trajectories = kapture.Trajectories()
    for trajectories, rig_mapping, sensor_mapping in zip(trajectories_list, rig_mappings, sensor_mappings):
        if trajectories is None:
            continue
        for timestamp, sensor_id, pose in kapture.flatten(trajectories):
            if sensor_id in rig_mapping:
                new_sensor_id = rig_mapping[sensor_id]
            else:
                new_sensor_id = sensor_mapping[sensor_id]
            merged_trajectories[(timestamp, new_sensor_id)] = pose
    return merged_trajectories


def merge_records_camera(
        records_camera_list: List[Optional[kapture.RecordsCamera]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsCamera:
    """
    Merge several camera records list into one list with new identifiers for the sensors.

    :param records_camera_list: list of camera records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged camera records
    """
    return merge_table_key2(
        table_list=records_camera_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsCamera
    )


def merge_records_lidar(
        records_lidar_list: List[Optional[kapture.RecordsLidar]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsLidar:
    """
    Merge several lidar records list into one list with new identifiers for the sensors.

    :param records_lidar_list: list of lidar records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged lidar records
    """
    return merge_table_key2(
        table_list=records_lidar_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsLidar
    )


def merge_records_wifi(
        records_wifi_list: List[Optional[kapture.RecordsWifi]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsWifi:
    """
    Merge several wifi records list into one list with new identifiers for the sensors.

    :param records_wifi_list: list of wifi records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged wifi records
    """
    return merge_table_key3(
        table_list=records_wifi_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsWifi,
        subdict_constructor=kapture.RecordsWifi.record_type
    )


def merge_records_bluetooth(
        records_bluetooth_list: List[Optional[kapture.RecordsBluetooth]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsBluetooth:
    """
    Merge several bluetooth records list into one list with new identifiers for the sensors.

    :param records_bluetooth_list: list of wifi records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged bluetooth records
    """
    return merge_table_key3(
        table_list=records_bluetooth_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsBluetooth,
        subdict_constructor=kapture.RecordsBluetooth.record_type
    )


def merge_records_gnss(
        records_gnss_list: List[Optional[kapture.RecordsGnss]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsGnss:
    """
    Merge several gnss records list into one list with new identifiers for the sensors.

    :param records_gnss_list: list of gnss records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged gnss records
    """
    return merge_table_key2(
        table_list=records_gnss_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsGnss
    )


def merge_records_accelerometer(
        records_accelerometer_list: List[Optional[kapture.RecordsAccelerometer]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsAccelerometer:
    """
    Merge several accelerometer records list into one list with new identifiers for the sensors.

    :param records_accelerometer_list: list of accelerometer records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged accelerometer records
    """
    return merge_table_key2(
        table_list=records_accelerometer_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsAccelerometer
    )


def merge_records_gyroscope(
        records_gyroscope_list: List[Optional[kapture.RecordsGyroscope]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsGyroscope:
    """
    Merge several gyroscope records list into one list with new identifiers for the sensors.

    :param records_gyroscope_list: list of gyroscope records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged gyroscope records
    """
    return merge_table_key2(
        table_list=records_gyroscope_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsGyroscope
    )


def merge_records_magnetic(
        records_magnetic_list: List[Optional[kapture.RecordsMagnetic]],
        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsMagnetic:
    """
    Merge several magnetic records list into one list with new identifiers for the sensors.

    :param records_magnetic_list: list of magnetic records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged magnetic records
    """
    return merge_table_key2(
        table_list=records_magnetic_list,
        sensor_mappings=sensor_mappings,
        table_constructor=kapture.RecordsMagnetic
    )


def merge_remap(kapture_list: List[kapture.Kapture],  # noqa: C901: function a bit long but not too complex
                skip_list: List[Type],
                data_paths: List[str],
                kapture_path: str,
                images_import_method: TransferAction) -> kapture.Kapture:
    """
    Merge multiple kapture while keeping ids (sensor_id) identical in merged and inputs.

    :param kapture_list: list of kapture to merge.
    :param skip_list: input optional types to not merge. sensors and rigs are unskippable
    :param data_paths: list of path to root path directory in same order as mentioned in kapture_list.
    :param kapture_path: directory root path to the merged kapture.
    :param images_import_method: method to transfer image files
    :return: merged kapture object
    """
    merged_kapture = kapture.Kapture()

    # find new sensor ids / rig ids
    sensors_mapping = []
    rigs_mapping = []
    _compute_new_ids(kapture_list, rigs_mapping, sensors_mapping)

    # concatenate all sensors with the remapped ids
    new_sensors = merge_sensors([a_kapture.sensors for a_kapture in kapture_list], sensors_mapping)
    # if merge_sensors returned an empty object, keep merged_kapture.sensors to None
    merged_kapture.sensors = get_new_if_not_empty(new_sensors, merged_kapture.sensors)

    # concatenate all rigs with the remapped ids
    new_rigs = merge_rigs([a_kapture.rigs for a_kapture in kapture_list], rigs_mapping, sensors_mapping)
    # if merge_rigs returned an empty object, keep merged_kapture.rigs to None
    merged_kapture.rigs = get_new_if_not_empty(new_rigs, merged_kapture.rigs)

    # all fields below can be skipped with skip_list
    # we do not assign the properties when the merge evaluate to false, we keep it as None
    if kapture.Trajectories not in skip_list:
        new_trajectories = merge_trajectories([a_kapture.trajectories for a_kapture in kapture_list],
                                              rigs_mapping,
                                              sensors_mapping)
        merged_kapture.trajectories = get_new_if_not_empty(new_trajectories, merged_kapture.trajectories)

    if kapture.RecordsCamera not in skip_list:
        new_records_camera = merge_records_camera([a_kapture.records_camera for a_kapture in kapture_list],
                                                  sensors_mapping)
        merged_kapture.records_camera = get_new_if_not_empty(new_records_camera, merged_kapture.records_camera)

        merge_records_data([[image_name
                             for _, _, image_name in kapture.flatten(every_kapture.records_camera)]
                            if every_kapture.records_camera is not None else []
                            for every_kapture in kapture_list],
                           [get_image_fullpath(data_path, image_filename=None) for data_path in data_paths],
                           kapture_path,
                           images_import_method)
    if kapture.RecordsLidar not in skip_list:
        new_records_lidar = merge_records_lidar([a_kapture.records_lidar for a_kapture in kapture_list],
                                                sensors_mapping)
        merged_kapture.records_lidar = get_new_if_not_empty(new_records_lidar, merged_kapture.records_lidar)
    if kapture.RecordsWifi not in skip_list:
        new_records_wifi = merge_records_wifi([a_kapture.records_wifi for a_kapture in kapture_list],
                                              sensors_mapping)
        merged_kapture.records_wifi = get_new_if_not_empty(new_records_wifi, merged_kapture.records_wifi)
    if kapture.RecordsBluetooth not in skip_list:
        new_records_bluetooth = merge_records_bluetooth([a_kapture.records_bluetooth for a_kapture in kapture_list],
                                                        sensors_mapping)
        merged_kapture.records_bluetooth = get_new_if_not_empty(new_records_bluetooth, merged_kapture.records_bluetooth)
    if kapture.RecordsGnss not in skip_list:
        new_records_gnss = merge_records_gnss([a_kapture.records_gnss for a_kapture in kapture_list],
                                              sensors_mapping)
        merged_kapture.records_gnss = get_new_if_not_empty(new_records_gnss,
                                                           merged_kapture.records_gnss)
    if kapture.RecordsAccelerometer not in skip_list:
        new_records_accelerometer = merge_records_accelerometer(
            [a_kapture.records_accelerometer for a_kapture in kapture_list],
            sensors_mapping)
        merged_kapture.records_accelerometer = get_new_if_not_empty(new_records_accelerometer,
                                                                    merged_kapture.records_accelerometer)
    if kapture.RecordsGyroscope not in skip_list:
        new_records_gyroscope = merge_records_gyroscope(
            [a_kapture.records_gyroscope for a_kapture in kapture_list],
            sensors_mapping)
        merged_kapture.records_gyroscope = get_new_if_not_empty(new_records_gyroscope,
                                                                merged_kapture.records_gyroscope)
    if kapture.RecordsMagnetic not in skip_list:
        new_records_magnetic = merge_records_magnetic(
            [a_kapture.records_magnetic for a_kapture in kapture_list],
            sensors_mapping)
        merged_kapture.records_magnetic = get_new_if_not_empty(new_records_magnetic,
                                                               merged_kapture.records_magnetic)

    # for the reconstruction, except points and observations, the files are copied with shutil.copy
    # if kapture_path evaluates to False, all copies will be skipped (but classes will be filled normally)
    if kapture.Keypoints not in skip_list:
        keypoints = [a_kapture.keypoints for a_kapture in kapture_list]
        keypoints_not_none = [k for k in keypoints if k is not None]
        if len(keypoints_not_none) > 0:
            new_keypoints = merge_keypoints(keypoints, data_paths, kapture_path)
            merged_kapture.keypoints = get_new_if_not_empty(new_keypoints, merged_kapture.keypoints)
    if kapture.Descriptors not in skip_list:
        descriptors = [a_kapture.descriptors for a_kapture in kapture_list]
        descriptors_not_none = [k for k in descriptors if k is not None]
        if len(descriptors_not_none) > 0:
            new_descriptors = merge_descriptors(descriptors, data_paths, kapture_path)
            merged_kapture.descriptors = get_new_if_not_empty(new_descriptors, merged_kapture.descriptors)
    if kapture.GlobalFeatures not in skip_list:
        global_features = [a_kapture.global_features for a_kapture in kapture_list]
        global_features_not_none = [k for k in global_features if k is not None]
        if len(global_features_not_none) > 0:
            new_global_features = merge_global_features(global_features, data_paths, kapture_path)
            merged_kapture.global_features = get_new_if_not_empty(new_global_features, merged_kapture.global_features)
    if kapture.Matches not in skip_list:
        matches = [a_kapture.matches for a_kapture in kapture_list]
        new_matches = merge_matches(matches, data_paths, kapture_path)
        merged_kapture.matches = get_new_if_not_empty(new_matches, merged_kapture.matches)

    if kapture.Points3d not in skip_list and kapture.Observations not in skip_list:
        points_and_obs = [(a_kapture.points3d, a_kapture.observations) for a_kapture in kapture_list]
        new_points, new_observations = merge_points3d_and_observations(points_and_obs)
        merged_kapture.points3d = get_new_if_not_empty(new_points, merged_kapture.points3d)
        merged_kapture.observations = get_new_if_not_empty(new_observations, merged_kapture.observations)
    elif kapture.Points3d not in skip_list:
        points = [a_kapture.points3d for a_kapture in kapture_list]
        new_points = merge_points3d(points)
        merged_kapture.points3d = get_new_if_not_empty(new_points, merged_kapture.points3d)
    return merged_kapture


def _compute_new_ids(kapture_list, rigs_mapping, sensors_mapping):
    sensor_offset = 0
    rigs_offset = 0
    for every_kapture in kapture_list:
        if every_kapture.sensors is not None:
            sensors_mapping.append(get_sensors_mapping(every_kapture.sensors, sensor_offset))
            sensor_offset += len(every_kapture.sensors)
        else:
            sensors_mapping.append({})

        if every_kapture.rigs is not None:
            rigs_mapping.append(get_rigs_mapping(every_kapture.rigs, rigs_offset))
            rigs_offset += len(every_kapture.rigs)
        else:
            rigs_mapping.append({})
