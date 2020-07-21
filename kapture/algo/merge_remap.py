# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Merge kapture objects with remapping of identifiers.
"""

from typing import List, Optional, Type, Dict

import kapture
from kapture.io.records import TransferAction, get_image_fullpath

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


def merge_sensors(sensors_list: List[Optional[kapture.Sensors]],
                  sensor_mappings: List[Dict[str, str]]) -> kapture.Sensors:
    """
    Merge several sensors list into one list with new identifiers.

    :param sensors_list: list of sensors definitions to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged sensors definitions
    """
    assert len(sensors_list) > 0
    assert len(sensors_list) == len(sensor_mappings)

    merged_sensors = kapture.Sensors()
    for sensors, mapping in zip(sensors_list, sensor_mappings):
        if sensors is None:
            continue
        for sensor_id in sensors.keys():
            new_id = mapping[sensor_id]
            merged_sensors[new_id] = sensors[sensor_id]
    return merged_sensors


def merge_rigs(rigs_list: List[Optional[kapture.Rigs]],
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


def merge_trajectories(trajectories_list: List[Optional[kapture.Trajectories]],
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


def merge_records_camera(records_camera_list: List[Optional[kapture.RecordsCamera]],
                         sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsCamera:
    """
    Merge several camera records list into one list with new identifiers for the sensors.

    :param records_camera_list: list of camera records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged camera records
    """
    assert len(records_camera_list) > 0
    assert len(records_camera_list) == len(sensor_mappings)

    merged_records_camera = kapture.RecordsCamera()
    for records_camera, sensor_mapping in zip(records_camera_list, sensor_mappings):
        if records_camera is None:
            continue
        for timestamp, sensor_id, filename in kapture.flatten(records_camera):
            new_sensor_id = sensor_mapping[sensor_id]
            merged_records_camera[(timestamp, new_sensor_id)] = filename
    return merged_records_camera


def merge_records_lidar(records_lidar_list: List[Optional[kapture.RecordsLidar]],
                        sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsLidar:
    """
    Merge several lidar records list into one list with new identifiers for the sensors.

    :param records_lidar_list: list of lidar records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged lidar records
    """
    assert len(records_lidar_list) > 0
    assert len(records_lidar_list) == len(sensor_mappings)

    merged_records_lidar = kapture.RecordsLidar()
    for records_lidar, sensor_mapping in zip(records_lidar_list, sensor_mappings):
        if records_lidar is None:
            continue
        for timestamp, sensor_id, filename in kapture.flatten(records_lidar):
            new_sensor_id = sensor_mapping[sensor_id]
            merged_records_lidar[(timestamp, new_sensor_id)] = filename
    return merged_records_lidar


def merge_records_wifi(records_wifi_list: List[Optional[kapture.RecordsWifi]],
                       sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsWifi:
    """
    Merge several wifi records list into one list with new identifiers for the sensors.

    :param records_wifi_list: list of wifi records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged wifi records
    """
    assert len(records_wifi_list) > 0
    assert len(records_wifi_list) == len(sensor_mappings)

    merged_wifi_records = kapture.RecordsWifi()
    for wifi_records, sensor_mapping in zip(records_wifi_list, sensor_mappings):
        if wifi_records is None:
            continue
        for timestamp, sensor_id, record_wifi in kapture.flatten(wifi_records):
            new_sensor_id = sensor_mapping[sensor_id]
            merged_wifi_records[(timestamp, new_sensor_id)] = record_wifi
    return merged_wifi_records


def merge_records_gnss(records_gnss_list: List[Optional[kapture.RecordsGnss]],
                       sensor_mappings: List[Dict[str, str]]) -> kapture.RecordsGnss:
    """
    Merge several gnss records list into one list with new identifiers for the sensors.

    :param records_gnss_list: list of gnss records to merge
    :param sensor_mappings: mapping of the sensor identifiers to their new identifiers
    :return: merged gnss records
    """
    assert len(records_gnss_list) > 0
    assert len(records_gnss_list) == len(sensor_mappings)

    merged_gnss_records = kapture.RecordsGnss()
    for gnss_records, sensor_mapping in zip(records_gnss_list, sensor_mappings):
        if gnss_records is None:
            continue
        for timestamp, sensor_id, record_gnss in kapture.flatten(gnss_records):
            new_sensor_id = sensor_mapping[sensor_id]
            merged_gnss_records[(timestamp, new_sensor_id)] = record_gnss
    return merged_gnss_records


def merge_remap(kapture_list: List[kapture.Kapture],
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
    :return:
    """
    merged_kapture = kapture.Kapture()

    # find new sensor ids / rig ids
    sensor_offset = 0
    rigs_offset = 0
    sensors_mapping = []
    rigs_mapping = []
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

    # concatenate all sensors with the remapped ids
    new_sensors = merge_sensors([a_kapture.sensors for a_kapture in kapture_list], sensors_mapping)
    if new_sensors:  # if merge_sensors returned an empty object, keep merged_kapture.sensors to None
        merged_kapture.sensors = new_sensors

    # concatenate all rigs with the remapped ids
    new_rigs = merge_rigs([a_kapture.rigs for a_kapture in kapture_list], rigs_mapping, sensors_mapping)
    if new_rigs:  # if merge_rigs returned an empty object, keep merged_kapture.rigs to None
        merged_kapture.rigs = new_rigs

    # all fields below can be skipped with skip_list
    # we do not assign the properties when the merge evaluate to false, we keep it as None
    if kapture.Trajectories not in skip_list:
        new_trajectories = merge_trajectories([a_kapture.trajectories for a_kapture in kapture_list],
                                              rigs_mapping,
                                              sensors_mapping)
        if new_trajectories:
            merged_kapture.trajectories = new_trajectories

    if kapture.RecordsCamera not in skip_list:
        new_records_camera = merge_records_camera([a_kapture.records_camera for a_kapture in kapture_list],
                                                  sensors_mapping)
        if new_records_camera:
            merged_kapture.records_camera = new_records_camera

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
        if new_records_lidar:
            merged_kapture.records_lidar = new_records_lidar
    if kapture.RecordsWifi not in skip_list:
        new_records_wifi = merge_records_wifi([a_kapture.records_wifi for a_kapture in kapture_list],
                                              sensors_mapping)
        if new_records_wifi:
            merged_kapture.records_wifi = new_records_wifi
    if kapture.RecordsGnss not in skip_list:
        new_records_gnss = merge_records_gnss([a_kapture.records_gnss for a_kapture in kapture_list],
                                              sensors_mapping)
        if new_records_gnss:
            merged_kapture.records_gnss = new_records_gnss

    # for the reconstruction, except points and observations, the files are copied with shutil.copy
    # if kapture_path evaluates to False, all copies will be skipped (but classes will be filled normally)
    if kapture.Keypoints not in skip_list:
        keypoints = [a_kapture.keypoints for a_kapture in kapture_list]
        keypoints_not_none = [k for k in keypoints if k is not None]
        if len(keypoints_not_none) > 0:
            new_keypoints = merge_keypoints(keypoints, data_paths, kapture_path)
            if new_keypoints:
                merged_kapture.keypoints = new_keypoints
    if kapture.Descriptors not in skip_list:
        descriptors = [a_kapture.descriptors for a_kapture in kapture_list]
        descriptors_not_none = [k for k in descriptors if k is not None]
        if len(descriptors_not_none) > 0:
            new_descriptors = merge_descriptors(descriptors, data_paths, kapture_path)
            if new_descriptors:
                merged_kapture.descriptors = new_descriptors
    if kapture.GlobalFeatures not in skip_list:
        global_features = [a_kapture.global_features for a_kapture in kapture_list]
        global_features_not_none = [k for k in global_features if k is not None]
        if len(global_features_not_none) > 0:
            new_global_features = merge_global_features(global_features, data_paths, kapture_path)
            if new_global_features:
                merged_kapture.global_features = new_global_features
    if kapture.Matches not in skip_list:
        matches = [a_kapture.matches for a_kapture in kapture_list]
        new_matches = merge_matches(matches, data_paths, kapture_path)
        if new_matches:
            merged_kapture.matches = new_matches

    if kapture.Points3d not in skip_list and kapture.Observations not in skip_list:
        points_and_obs = [(a_kapture.points3d, a_kapture.observations) for a_kapture in kapture_list]
        new_points, new_observations = merge_points3d_and_observations(points_and_obs)
        if new_points:
            merged_kapture.points3d = new_points
        if new_observations:
            merged_kapture.observations = new_observations
    elif kapture.Points3d not in skip_list:
        points = [a_kapture.points3d for a_kapture in kapture_list]
        new_points = merge_points3d(points)
        if new_points:
            merged_kapture.points3d = new_points
    return merged_kapture
