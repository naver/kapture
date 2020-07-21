# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Merge kapture objects.
"""

from typing import List, Optional, Type

import kapture
from kapture.io.records import TransferAction, get_image_fullpath

from .merge_reconstruction import merge_keypoints, merge_descriptors, merge_global_features, merge_matches
from .merge_reconstruction import merge_points3d_and_observations, merge_points3d
from .merge_records_data import merge_records_data


def merge_sensors(sensors_list: List[Optional[kapture.Sensors]]) -> kapture.Sensors:
    """
    Merge several sensors lists. For sensor with the same identifier, keep only the first one.

    :param sensors_list: list of sensors
    :return: merge sensors
    """
    assert len(sensors_list) > 0

    merged_sensors = kapture.Sensors()
    for sensors in sensors_list:
        if sensors is None:
            continue
        for sensor_id in sensors.keys():
            if sensor_id in merged_sensors:
                continue
            merged_sensors[sensor_id] = sensors[sensor_id]
    return merged_sensors


def merge_rigs(rigs_list: List[Optional[kapture.Rigs]]) -> kapture.Rigs:
    """
    Merge several rigs lists. For sensor with the same rig and sensor identifier, keep only the first one.

    :param rigs_list: list of rigs
    :return: merged rigs
    """
    assert len(rigs_list) > 0

    merged_rigs = kapture.Rigs()
    for rigs in rigs_list:
        if rigs is None:
            continue
        for rig_id, sensor_id in rigs.key_pairs():
            if rig_id in merged_rigs and sensor_id in merged_rigs[rig_id]:
                continue
            merged_rigs[(rig_id, sensor_id)] = rigs[(rig_id, sensor_id)]
    return merged_rigs


def merge_trajectories(trajectories_list: List[Optional[kapture.Trajectories]]) -> kapture.Trajectories:
    """
    Merge several trajectories lists. For trajectory point with the same timestamp and sensor identifier,
     keep only the first one.

    :param trajectories_list: list of trajectories
    :return: merged trajectories
    """
    assert len(trajectories_list) > 0

    merged_trajectories = kapture.Trajectories()
    for trajectories in trajectories_list:
        if trajectories is None:
            continue
        for timestamp, sensor_id, pose in kapture.flatten(trajectories):
            if (timestamp, sensor_id) in merged_trajectories:
                continue
            merged_trajectories[(timestamp, sensor_id)] = pose
    return merged_trajectories


def merge_records_camera(records_camera_list: List[Optional[kapture.RecordsCamera]]) -> kapture.RecordsCamera:
    """
    Merge several camera records lists. For camera record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_camera_list: list of camera records
    :return: merged camera records
    """
    assert len(records_camera_list) > 0

    merged_records_camera = kapture.RecordsCamera()
    for records_camera in records_camera_list:
        if records_camera is None:
            continue
        for timestamp, sensor_id, filename in kapture.flatten(records_camera):
            if (timestamp, sensor_id) in merged_records_camera:
                continue
            merged_records_camera[(timestamp, sensor_id)] = filename
    return merged_records_camera


def merge_records_lidar(records_lidar_list: List[Optional[kapture.RecordsLidar]]) -> kapture.RecordsLidar:
    """
    Merge several lidar records lists. For lidar record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_lidar_list: list of lidar records
    :return: merged lidar records
    """
    assert len(records_lidar_list) > 0

    merged_records_lidar = kapture.RecordsLidar()
    for records_lidar in records_lidar_list:
        if records_lidar is None:
            continue
        for timestamp, sensor_id, filename in kapture.flatten(records_lidar):
            if (timestamp, sensor_id) in merged_records_lidar:
                continue
            merged_records_lidar[(timestamp, sensor_id)] = filename
    return merged_records_lidar


def merge_records_wifi(records_wifi_list: List[Optional[kapture.RecordsWifi]]) -> kapture.RecordsWifi:
    """
    Merge several wifi records lists. For wifi record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_wifi_list: list of wifi records
    :return: merged wifi records
    """
    assert len(records_wifi_list) > 0

    merged_records_wifi = kapture.RecordsWifi()
    for records_wifi in records_wifi_list:
        if records_wifi is None:
            continue
        for timestamp, sensor_id, record_wifi in kapture.flatten(records_wifi):
            if (timestamp, sensor_id) in merged_records_wifi:
                continue
            merged_records_wifi[(timestamp, sensor_id)] = record_wifi
    return merged_records_wifi


def merge_records_gnss(records_gnss_list: List[Optional[kapture.RecordsGnss]]) -> kapture.RecordsGnss:
    """
    Merge several gnss records lists. For gnss record with the same timestamp and sensor identifier,
     keep only the first one.

    :param records_gnss_list: list of gnss records
    :return: merged gnss records
    """
    assert len(records_gnss_list) > 0

    merged_records_gnss = kapture.RecordsWifi()
    for records_gnss in records_gnss_list:
        if records_gnss is None:
            continue
        for timestamp, sensor_id, record_gnss in kapture.flatten(records_gnss):
            if (timestamp, sensor_id) in merged_records_gnss:
                continue
            merged_records_gnss[(timestamp, sensor_id)] = record_gnss
    return merged_records_gnss


def merge_keep_ids(kapture_list: List[kapture.Kapture], skip_list: List[Type],
                   data_paths: List[str], kapture_path: str,
                   images_import_method: TransferAction) -> kapture.Kapture:
    """
    Merge multiple kapture while keeping ids (sensor_id) identical in merged and inputs.

    :param kapture_list: list of kapture to merge.
    :param skip_list: optional types not to merge. sensors and rigs are unskipable
    :param data_paths: list of path to root path directory in same order as mentioned in kapture_list.
    :param kapture_path: directory root path to the merged kapture.
    :return: merged kapture
    """
    merged_kapture = kapture.Kapture()

    # get the union of all sensors
    new_sensors = merge_sensors([every_kapture.sensors for every_kapture in kapture_list])
    if new_sensors:  # if merge_sensors returned an empty object, keep merged_kapture.sensors to None
        merged_kapture.sensors = new_sensors

    # get the union of all rigs
    new_rigs = merge_rigs([every_kapture.rigs for every_kapture in kapture_list])
    if new_rigs:  # if merge_rigs returned an empty object, keep merged_kapture.sensors to None
        merged_kapture.rigs = new_rigs

    # all fields below can be skipped with skip_list
    # we do not assign the properties when the merge evaluate to false, we keep it as None
    if kapture.Trajectories not in skip_list:
        new_trajectories = merge_trajectories([every_kapture.trajectories for every_kapture in kapture_list])
        if new_trajectories:
            merged_kapture.trajectories = new_trajectories

    if kapture.RecordsCamera not in skip_list:
        new_records_camera = merge_records_camera([every_kapture.records_camera for every_kapture in kapture_list])
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
        new_records_lidar = merge_records_lidar([every_kapture.records_lidar for every_kapture in kapture_list])
        if new_records_lidar:
            merged_kapture.records_lidar = new_records_lidar
    if kapture.RecordsWifi not in skip_list:
        new_records_wifi = merge_records_wifi([every_kapture.records_wifi for every_kapture in kapture_list])
        if new_records_wifi:
            merged_kapture.records_wifi = new_records_wifi
    if kapture.RecordsGnss not in skip_list:
        new_records_gnss = merge_records_gnss([every_kapture.records_gnss for every_kapture in kapture_list])
        if new_records_gnss:
            merged_kapture.records_gnss = new_records_gnss

    # for the reconstruction, except points and observations, the files are copied with shutil.copy
    # if kapture_path evaluates to False, all copies will be skipped (but classes will be filled normally)
    if kapture.Keypoints not in skip_list:
        keypoints = [every_kapture.keypoints for every_kapture in kapture_list]
        keypoints_not_none = [k for k in keypoints if k is not None]
        if len(keypoints_not_none) > 0:
            new_keypoints = merge_keypoints(keypoints, data_paths, kapture_path)
            if new_keypoints:
                merged_kapture.keypoints = new_keypoints
    if kapture.Descriptors not in skip_list:
        descriptors = [every_kapture.descriptors for every_kapture in kapture_list]
        descriptors_not_none = [k for k in descriptors if k is not None]
        if len(descriptors_not_none) > 0:
            new_descriptors = merge_descriptors(descriptors, data_paths, kapture_path)
            if new_descriptors:
                merged_kapture.descriptors = new_descriptors
    if kapture.GlobalFeatures not in skip_list:
        global_features = [every_kapture.global_features for every_kapture in kapture_list]
        global_features_not_none = [k for k in global_features if k is not None]
        if len(global_features_not_none) > 0:
            new_global_features = merge_global_features(global_features, data_paths, kapture_path)
            if new_global_features:
                merged_kapture.global_features = new_global_features
    if kapture.Matches not in skip_list:
        matches = [every_kapture.matches for every_kapture in kapture_list]
        new_matches = merge_matches(matches, data_paths, kapture_path)
        if new_matches:
            merged_kapture.matches = new_matches

    if kapture.Points3d not in skip_list and kapture.Observations not in skip_list:
        points_and_obs = [(every_kapture.points3d, every_kapture.observations) for every_kapture in kapture_list]
        new_points, new_observations = merge_points3d_and_observations(points_and_obs)
        if new_points:
            merged_kapture.points3d = new_points
        if new_observations:
            merged_kapture.observations = new_observations
    elif kapture.Points3d not in skip_list:
        points = [every_kapture.points3d for every_kapture in kapture_list]
        new_points = merge_points3d(points)
        if new_points:
            merged_kapture.points3d = new_points
    return merged_kapture
