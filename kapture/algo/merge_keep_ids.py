# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Merge kapture objects.
"""

from typing import List, Optional, Type

import kapture
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.utils.Collections import get_new_if_not_empty

from .merge_reconstruction import merge_keypoints, merge_descriptors, merge_global_features, merge_matches
from .merge_reconstruction import merge_points3d_and_observations, merge_points3d
from .merge_records_data import merge_records_data


def merge_table_key1(
        table_list,
        table_constructor,
):
    """
    Merge several table with 1 key (eg. device_id)  into one.
    If multiple entry for a key keep only the first one.

    :param table_list: list of table to merge.
    :param table_constructor: the class type of table.
    :return table_merged
    """
    assert len(table_list) > 0
    table_list = [table for table in table_list if table is not None]
    if not all(isinstance(table, table_constructor) for table in table_list):
        raise TypeError('unexpected type.')
    table_merged = table_constructor()
    for table in table_list:
        for key1, entry in kapture.flatten(table):
            if key1 in table_merged:
                # skip to keep the first one
                continue
            table_merged[key1] = entry
    return table_merged


def merge_table_key2(
        table_list,
        table_constructor,
):
    """
    Merge several table with 2 keys (eg. timestamps, device_id)  into one.
    If multiple entry for a key keep only the first one.

    :param table_list: list of table to merge.
    :param table_constructor: the class type of table.
    :return table_merged
    """
    assert len(table_list) > 0
    table_list = [table for table in table_list if table is not None]
    if not all(isinstance(table, table_constructor) for table in table_list):
        raise TypeError('unexpected type.')
    table_merged = table_constructor()
    for table in table_list:
        for key1, key2, entry in kapture.flatten(table):
            if (key1, key2) in table_merged:
                # skip to keep the first one
                continue
            table_merged[key1, key2] = entry
    return table_merged


def merge_table_key3(
        table_list,
        table_constructor,
        subdict_constructor=dict,
):
    """
    Merge several records lists. Records is a dict (eg. wifi)).
    For record with the same timestamp and sensor identifier, keep only the first one.

    :param table_list: list of table to merge.
    :param table_constructor: the class type of table.
    :param subdict_constructor: used to create a new Dict type
    :return table_merged
    """
    assert len(table_list) > 0
    table_list = [table for table in table_list if table is not None]
    if not all(isinstance(table, table_constructor) for table in table_list):
        raise TypeError('unexpected type.')
    table_merged = table_constructor()
    for table in table_list:
        for key1, key2, key3, entry in kapture.flatten(table):
            if (key1, key2) not in table_merged:
                # if timestamp, sensor_id not there yet, create an instance of dict record
                table_merged[key1, key2] = subdict_constructor()
            table_merged[key1, key2].setdefault(key3, entry)
    return table_merged


def merge_sensors(
        sensors_list: List[Optional[kapture.Sensors]]
) -> kapture.Sensors:
    """
    Merge several sensors lists. For sensor with the same identifier, keep only the first one.

    :param sensors_list: list of sensors
    :return: merge sensors
    """
    return merge_table_key1(
        table_list=sensors_list,
        table_constructor=kapture.Sensors
    )


def merge_rigs(
        rigs_list: List[Optional[kapture.Rigs]]
) -> kapture.Rigs:
    """
    Merge several rigs lists. For sensor with the same rig and sensor identifier, keep only the first one.

    :param rigs_list: list of rigs
    :return: merged rigs
    """
    return merge_table_key2(
        table_list=rigs_list,
        table_constructor=kapture.Rigs
    )


def merge_trajectories(
        trajectories_list: List[Optional[kapture.Trajectories]]
) -> kapture.Trajectories:
    """
    Merge several trajectories lists. For trajectory point with the same timestamp and sensor identifier,
     keep only the first one.

    :param trajectories_list: list of trajectories
    :return: merged trajectories
    """
    return merge_table_key2(
        table_list=trajectories_list,
        table_constructor=kapture.Trajectories
    )


def merge_records_camera(
        records_camera_list: List[Optional[kapture.RecordsCamera]]
) -> kapture.RecordsCamera:
    """
    Merge several camera records lists. For camera record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_camera_list: list of camera records
    :return: merged camera records
    """
    return merge_table_key2(
        table_list=records_camera_list,
        table_constructor=kapture.RecordsCamera
    )


def merge_records_lidar(
        records_lidar_list: List[Optional[kapture.RecordsLidar]]
) -> kapture.RecordsLidar:
    """
    Merge several lidar records lists. For lidar record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_lidar_list: list of lidar records
    :return: merged lidar records
    """
    return merge_table_key2(
        table_list=records_lidar_list,
        table_constructor=kapture.RecordsLidar
    )


def merge_records_wifi(
        records_wifi_list: List[Optional[kapture.RecordsWifi]]
) -> kapture.RecordsWifi:
    """
    Merge several wifi records lists.
    For wifi record with the same timestamp, sensor, BSSID,
     keep only the first one.

    :param records_wifi_list: list of wifi records
    :return: merged wifi records
    """
    return merge_table_key3(
        table_list=records_wifi_list,
        table_constructor=kapture.RecordsWifi,
        subdict_constructor=kapture.RecordsWifi.record_type
    )


def merge_records_bluetooth(
        records_bluetooth_list: List[Optional[kapture.RecordsBluetooth]]
) -> kapture.RecordsBluetooth:
    """
    Merge several bluetooth records lists.
    For bluetooth record with the same timestamp, sensor, address,
     keep only the first one.

    :param records_bluetooth_list: list of wifi records
    :return: merged bluetooth records
    """
    return merge_table_key3(
        table_list=records_bluetooth_list,
        table_constructor=kapture.RecordsBluetooth,
        subdict_constructor=kapture.RecordsBluetooth.record_type
    )


def merge_records_gnss(
        records_gnss_list: List[Optional[kapture.RecordsGnss]]
) -> kapture.RecordsGnss:
    """
    Merge several gnss records lists. For gnss record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_gnss_list: list of gnss records
    :return: merged gnss records
    """
    return merge_table_key2(
        table_list=records_gnss_list,
        table_constructor=kapture.RecordsGnss
    )


def merge_records_accelerometer(
        records_accelerometer_list: List[Optional[kapture.RecordsAccelerometer]]
) -> kapture.RecordsAccelerometer:
    """
    Merge several accelerometer records lists.
    For accelerometer record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_accelerometer_list: list of accelerometer records
    :return: merged accelerometer records
    """
    return merge_table_key2(
        table_list=records_accelerometer_list,
        table_constructor=kapture.RecordsAccelerometer
    )


def merge_records_gyroscope(
        records_gyroscope_list: List[Optional[kapture.RecordsGyroscope]]
) -> kapture.RecordsGyroscope:
    """
    Merge several gyroscope records lists.
    For gyroscope record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_gyroscope_list: list of gnss records
    :return: merged gyroscope records
    """
    return merge_table_key2(
        table_list=records_gyroscope_list,
        table_constructor=kapture.RecordsGyroscope
    )


def merge_records_magnetic(
        records_magnetic_list: List[Optional[kapture.RecordsMagnetic]]
) -> kapture.RecordsMagnetic:
    """
    Merge several magnetic records lists.
    For magnetic record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_magnetic_list: list of gnss records
    :return: merged magnetic records
    """
    return merge_table_key2(
        table_list=records_magnetic_list,
        table_constructor=kapture.RecordsMagnetic
    )


def merge_keep_ids(kapture_list: List[kapture.Kapture],  # noqa: C901: function a bit long but not too complex
                   skip_list: List[Type],
                   data_paths: List[str],
                   kapture_path: str,
                   images_import_method: TransferAction) -> kapture.Kapture:
    """
    Merge multiple kapture while keeping ids (sensor_id) identical in merged and inputs.

    :param kapture_list: list of kapture to merge.
    :param skip_list: optional types not to merge. sensors and rigs are unskippable
    :param data_paths: list of path to root path directory in same order as mentioned in kapture_list.
    :param kapture_path: directory root path to the merged kapture.
    :param images_import_method: method to transfer image files
    :return: merged kapture
    """
    merged_kapture = kapture.Kapture()

    # get the union of all sensors
    new_sensors = merge_sensors([every_kapture.sensors for every_kapture in kapture_list])
    # if merge_sensors returned an empty object, keep merged_kapture.sensors to None
    merged_kapture.sensors = get_new_if_not_empty(new_sensors, merged_kapture.sensors)

    # get the union of all rigs
    new_rigs = merge_rigs([every_kapture.rigs for every_kapture in kapture_list])
    # if merge_rigs returned an empty object, keep merged_kapture.rigs to None
    merged_kapture.rigs = get_new_if_not_empty(new_rigs, merged_kapture.rigs)

    # all fields below can be skipped with skip_list
    # we do not assign the properties when the merge evaluate to false, we keep it as None
    if kapture.Trajectories not in skip_list:
        new_trajectories = merge_trajectories([every_kapture.trajectories for every_kapture in kapture_list])
        merged_kapture.trajectories = get_new_if_not_empty(new_trajectories, merged_kapture.trajectories)

    if kapture.RecordsCamera not in skip_list:
        new_records_camera = merge_records_camera([every_kapture.records_camera for every_kapture in kapture_list])
        merged_kapture.records_camera = get_new_if_not_empty(new_records_camera, merged_kapture.records_camera)

        merge_records_data([[image_name
                             for _, _, image_name in kapture.flatten(every_kapture.records_camera)]
                            if every_kapture.records_camera is not None else []
                            for every_kapture in kapture_list],
                           [get_image_fullpath(data_path, image_filename=None) for data_path in data_paths],
                           kapture_path,
                           images_import_method)

    if kapture.RecordsLidar not in skip_list:
        new_records_lidar = merge_records_lidar([every_kapture.records_lidar
                                                 for every_kapture in kapture_list])
        merged_kapture.records_lidar = get_new_if_not_empty(new_records_lidar,
                                                            merged_kapture.records_lidar)
    if kapture.RecordsWifi not in skip_list:
        new_records_wifi = merge_records_wifi([every_kapture.records_wifi
                                               for every_kapture in kapture_list])
        merged_kapture.records_wifi = get_new_if_not_empty(new_records_wifi,
                                                           merged_kapture.records_wifi)
    if kapture.RecordsBluetooth not in skip_list:
        new_records_bluetooth = merge_records_bluetooth([every_kapture.records_bluetooth
                                                         for every_kapture in kapture_list])
        merged_kapture.records_bluetooth = get_new_if_not_empty(new_records_bluetooth,
                                                                merged_kapture.records_bluetooth)
    if kapture.RecordsGnss not in skip_list:
        new_records_gnss = merge_records_gnss([every_kapture.records_gnss
                                               for every_kapture in kapture_list])
        merged_kapture.records_gnss = get_new_if_not_empty(new_records_gnss,
                                                           merged_kapture.records_gnss)
    if kapture.RecordsAccelerometer not in skip_list:
        new_records_accelerometer = merge_records_accelerometer([every_kapture.records_accelerometer
                                                                 for every_kapture in kapture_list])
        merged_kapture.records_accelerometer = get_new_if_not_empty(new_records_accelerometer,
                                                                    merged_kapture.records_accelerometer)
    if kapture.RecordsGyroscope not in skip_list:
        new_records_gyroscope = merge_records_gyroscope([every_kapture.records_gyroscope
                                                         for every_kapture in kapture_list])
        merged_kapture.records_gyroscope = get_new_if_not_empty(new_records_gyroscope,
                                                                merged_kapture.records_gyroscope)
    if kapture.RecordsMagnetic not in skip_list:
        new_records_magnetic = merge_records_magnetic([every_kapture.records_magnetic
                                                       for every_kapture in kapture_list])
        merged_kapture.records_magnetic = get_new_if_not_empty(new_records_magnetic,
                                                               merged_kapture.records_magnetic)

    # for the reconstruction, except points and observations, the files are copied with shutil.copy
    # if kapture_path evaluates to False, all copies will be skipped (but classes will be filled normally)
    if kapture.Keypoints not in skip_list:
        keypoints = [every_kapture.keypoints for every_kapture in kapture_list]
        keypoints_not_none = [k for k in keypoints if k is not None]
        if len(keypoints_not_none) > 0:
            new_keypoints = merge_keypoints(keypoints, data_paths, kapture_path)
            merged_kapture.keypoints = get_new_if_not_empty(new_keypoints, merged_kapture.keypoints)
    if kapture.Descriptors not in skip_list:
        descriptors = [every_kapture.descriptors for every_kapture in kapture_list]
        descriptors_not_none = [k for k in descriptors if k is not None]
        if len(descriptors_not_none) > 0:
            new_descriptors = merge_descriptors(descriptors, data_paths, kapture_path)
            merged_kapture.descriptors = get_new_if_not_empty(new_descriptors, merged_kapture.descriptors)
    if kapture.GlobalFeatures not in skip_list:
        global_features = [every_kapture.global_features for every_kapture in kapture_list]
        global_features_not_none = [k for k in global_features if k is not None]
        if len(global_features_not_none) > 0:
            new_global_features = merge_global_features(global_features, data_paths, kapture_path)
            merged_kapture.global_features = get_new_if_not_empty(new_global_features, merged_kapture.global_features)
    if kapture.Matches not in skip_list:
        matches = [every_kapture.matches for every_kapture in kapture_list]
        new_matches = merge_matches(matches, data_paths, kapture_path)
        merged_kapture.matches = get_new_if_not_empty(new_matches, merged_kapture.matches)

    if kapture.Points3d not in skip_list and kapture.Observations not in skip_list:
        points_and_obs = [(every_kapture.points3d, every_kapture.observations) for every_kapture in kapture_list]
        new_points, new_observations = merge_points3d_and_observations(points_and_obs)
        merged_kapture.points3d = get_new_if_not_empty(new_points, merged_kapture.points3d)
        merged_kapture.observations = get_new_if_not_empty(new_observations, merged_kapture.observations)
    elif kapture.Points3d not in skip_list:
        points = [every_kapture.points3d for every_kapture in kapture_list]
        new_points = merge_points3d(points)
        merged_kapture.points3d = get_new_if_not_empty(new_points, merged_kapture.points3d)
    return merged_kapture
