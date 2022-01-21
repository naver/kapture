# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
The trajectories points. We store for every timestamp and device a pose.
So you can have two different devices for the same timestamp with a different pose.
The timestamps are integers, and device identifiers strings.
The timestamps are often epoch timestamps (in micro or milliseconds), but can be any set of integers.
Note that the timestamps are usually ordered ascending (as the time goes), but this is not guaranteed,
and might change if you manipulate them, save and reload them from disk.
If you need them ordered, do it yourself.
"""

import logging
import quaternion
from tqdm import tqdm

from .PoseTransform import PoseTransform
from .Rigs import Rigs
from .flatten import flatten
from kapture.utils.logging import getLogger
import kapture.utils.computation as computation
from bisect import bisect_left
from copy import deepcopy
import sys
from typing import Union, Dict, List, Optional, Set, Tuple


class Trajectories(Dict[int, Dict[str, PoseTransform]]):
    """
    brief: Trajectories
            trajectories[timestamp][sensor_id] = <PoseTransform>
            or
            trajectories[(timestamp, sensor_id)] = <PoseTransform>

            with <PoseTransform> transforming points from world to device.
    """

    def __init__(self):
        super().__init__()
        self._timestamps_sorted_list = []
        self._first_timestamp = 0
        self._last_timestamp = sys.maxsize

    def __setitem__(self,
                    key: Union[int, Tuple[int, str]],
                    value: Union[Dict[str, PoseTransform], PoseTransform]):
        # enforce type checking
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            if not isinstance(value, PoseTransform):
                raise TypeError('invalid pose')
            self.setdefault(timestamp, {})[device_id] = value
            self._timestamps_sorted_list = []
        elif isinstance(key, int):
            # key is a timestamp
            timestamp = key
            # check type of values?
            if not isinstance(value, dict):
                raise TypeError('invalid value for trajectory timestamp')
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError('invalid device_id')
            if not all(isinstance(v, PoseTransform) for v in value.values()):
                raise TypeError('invalid Pose')
            super(Trajectories, self).__setitem__(timestamp, value)
            self._timestamps_sorted_list = []
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __getitem__(self, key: Union[int, Tuple[int, str]]) -> Union[Dict[str, PoseTransform], PoseTransform]:
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            return super(Trajectories, self).__getitem__(timestamp)[device_id]
        elif isinstance(key, int):
            # key is a timestamp
            return super(Trajectories, self).__getitem__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __delitem__(self, key: Union[int, Tuple[int, str]]):
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            super(Trajectories, self).__getitem__(timestamp).__delitem__(device_id)
            if len(super(Trajectories, self).__getitem__(timestamp)) == 0:
                # Cleaning upper level
                super(Trajectories, self).__delitem__(timestamp)
                self._timestamps_sorted_list = []
        elif isinstance(key, int):
            # key is a timestamp
            super(Trajectories, self).__delitem__(key)
            self._timestamps_sorted_list = []
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def timestamps_sorted_list(self) -> List[int]:
        """
        Get the list of timestamps is ascending sorted order
        """
        if len(self._timestamps_sorted_list) == 0:
            # Need to sort
            self._timestamps_sorted_list = sorted(list(self.keys()))
            if len(self._timestamps_sorted_list) > 0:
                self._first_timestamp = self._timestamps_sorted_list[0]
                if len(self._timestamps_sorted_list) > 1:
                    self._last_timestamp = self._timestamps_sorted_list[-1]
        return self._timestamps_sorted_list

    def timestamp_length(self) -> int:
        """
        Compute the trajectory timestamp length, which can be 10, 13 or 16 if these are epoch values, and anything.

        :return: Length of the timestamps as a positive integer, or -1 if it is variable
        """
        timestamps = self.timestamps_sorted_list()
        base_length = computation.num_digits(timestamps[0]) if len(timestamps) > 0 else -1
        indexes = [1, 2, 3, 4, 5, -1, -2, -3, -4] if len(timestamps) > 10 else list(range(1, len(timestamps)))
        for n in indexes:
            length = computation.num_digits(timestamps[n])
            if length != base_length:
                return -1
        return base_length

    def key_pairs(self) -> List[Tuple[int, str]]:
        """
        Returns the list of (timestamp, device_id) contained in trajectories.
        Those pairs can be used to access a single trajectory pose.
        :return: list of (timestamp, device_id)
        """
        return [
            (timestamp, sensor_id)
            for timestamp, sensors in self.items()
            for sensor_id in sensors.keys()
        ]

    @property
    def sensors_ids(self) -> Set[str]:
        """
        :return: the set of unique sensors identifiers in the trajectories
        """
        return set(
            sensor_id
            for timestamp, sensors in self.items()
            for sensor_id in sensors.keys()
        )

    def __contains__(self, key: Union[int, Tuple[int, str]]):
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            return super(Trajectories, self).__contains__(timestamp) and self[timestamp].__contains__(device_id)
        elif isinstance(key, int):
            return super(Trajectories, self).__contains__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __repr__(self) -> str:
        # [timestamp, sensor_id] = qw, qx, qy, qz, tx, ty, tz
        lines = [f'[ {timestamp:010}, {sensor_id:5}] = {pose}'
                 for timestamp, sensors in self.items()
                 for sensor_id, pose in sensors.items()]
        return '\n'.join(lines)

    def intermediate_pose(self, timestamp: int, device_id: str, max_interval: int) -> Optional[PoseTransform]:
        """
        Computes an intermediate pose in the trajectory of a device
        The timestamp should be in epoch precision
        i.e. 10, 13, 16 or 19 digits for seconds, milli-seconds, micro-seconds or nano-seconds
        and in the same precision as the trajectories timestamps themselves to work well.
        The timestamps interval should also be not too big (for example one second) to be of good value.
        The max_interval parameter value should be of the same scale as the timestamps.

        :param timestamp: timestamp
        :param device_id: device identifier
        :param max_interval: max interval between the given timestamp and the trajectory timestamps.
        :return: a compute 6D pose if found, None otherwise
        """
        if not isinstance(timestamp, int):
            raise TypeError('invalid timestamp')
        if not isinstance(device_id, str):
            raise TypeError('invalid device_id')
        # In case the pose already exist: just return it
        if self.__contains__(timestamp) and self.__getitem__(timestamp).__contains__(device_id):
            return self.__getitem__(timestamp).__getitem__(device_id)
        timestamps_with_poses_list = self.timestamps_sorted_list()
        # Check if the pose is out of bounds
        if timestamp <= self._first_timestamp or timestamp >= self._last_timestamp:
            return None
        # Find closest timestamps before and after
        next_position = bisect_left(timestamps_with_poses_list, timestamp)
        low_position = next_position - 1
        previous_ts = timestamps_with_poses_list[low_position]
        next_ts = timestamps_with_poses_list[next_position]
        # We should have found the two closest timestamps
        # Check there is a pose in the time interval for the device
        while timestamp - previous_ts <= max_interval and not self.__getitem__(previous_ts).__contains__(device_id):
            # Search backward for a timestamp with this device
            low_position -= 1
            if low_position >= 0:
                previous_ts = timestamps_with_poses_list[low_position]
            else:
                # We have reached the begin of the list without solution
                return None
        # Check the interval for the previous timestamp
        if timestamp - previous_ts > max_interval:
            # We are to far in the past
            return None
        while next_ts - timestamp <= max_interval and not self.__getitem__(next_ts).__contains__(device_id):
            # Search backward for a timestamp with this device
            next_position += 1
            if next_position < len(timestamps_with_poses_list):
                next_ts = timestamps_with_poses_list[next_position]
            else:
                # We have reached the end of the list without solution
                return None
        # Check the interval for the next timestamp
        if next_ts - timestamp > max_interval:
            # We are to far in the future
            return None
        previous_pose = self.__getitem__(previous_ts).__getitem__(device_id)
        next_pose = self.__getitem__(next_ts).__getitem__(device_id)
        return compute_intermediate_pose(timestamp, previous_ts, previous_pose, next_ts, next_pose)

    def inverse(self) -> 'Trajectories':
        """ :return: new trajectories with all pose inverted """
        trajectories_inverted = Trajectories()
        for timestamp, sensor_id in self.key_pairs():
            trajectories_inverted[timestamp, sensor_id] = self[timestamp, sensor_id].inverse()
        return trajectories_inverted


def rigs_remove(trajectories: Trajectories, rigs: Rigs) -> Trajectories:
    """
    Returns a new trajectories based on the given one,
    but where rigs poses are replaced by the poses of every sensors.
    This is useful for formats that does not have the rig notion.

    :param trajectories: input Trajectories where the rigs has to be replaced
    :param rigs: input Rigs that defines the rigs/sensors relationship.
    :return: output Trajectories where the rigs has been replaced.
    """
    assert isinstance(rigs, Rigs)
    assert isinstance(trajectories, Trajectories)
    new_trajectories = deepcopy(trajectories)
    rigs_remove_inplace(new_trajectories, rigs)
    return new_trajectories


def rigs_remove_inplace(trajectories: Trajectories, rigs: Rigs, max_depth: int = 10):
    """
    Removes rigs poses and replaces them by the poses of every sensors in it.
    The operation is performed inplace, and modifies trajectories.
    This is useful for formats that does not have the rig notion.

    :param trajectories: input/output Trajectories where the rigs has to be replaced
    :param rigs: input Rigs that defines the rigs/sensors relationship.
    :param max_depth: maximum nested rig depth.
    """
    assert isinstance(rigs, Rigs)
    assert isinstance(trajectories, Trajectories)
    # collect all poses of rigs in trajectories
    for iteration in range(max_depth):
        # repeat the operation while there is so rig remaining (nested rigs)
        jobs = [(timestamp, rig_id, pose_rig_from_world)
                for timestamp, poses_for_timestamp in trajectories.items()
                for rig_id, pose_rig_from_world in poses_for_timestamp.items()
                if rig_id in rigs]

        if len(jobs) == 0:
            # we are done
            break

        getLogger().debug(f'rigs_remove {len(jobs)} jobs at depth {iteration}')
        for timestamp, rig_id, pose_rig_from_world in tqdm(jobs, disable=getLogger().level >= logging.CRITICAL):
            for device_id, pose_device_from_rig in rigs[rig_id].items():
                pose_cam_from_world = PoseTransform.compose([pose_device_from_rig, pose_rig_from_world])
                trajectories.setdefault(timestamp, {})[device_id] = pose_cam_from_world
            del trajectories[timestamp][rig_id]

    for timestamp in trajectories.keys():
        if len(trajectories[timestamp]) == 0:
            del trajectories[timestamp]
    # Do not clear, the rigs : so easy to do outside, and so easy


def rigs_recover(
        trajectories: Trajectories,
        rigs: Rigs,
        master_sensors: Optional[List[str]] = None
) -> Trajectories:
    """
    Returns a new trajectories based on the given one,
    but where sensor poses are replaced by the rig pose, each time the sensor is registered in a rig.
    The rig configuration must be given.
    Warning: in some configuration, it may leads to inconsistent results. See code for details.

    :param trajectories: input Trajectories.
    :param rigs: input Rigs configuration.
    :param master_sensors: input If given, only compute rig poses for the given sensors.
    :return: a new Trajectories.
    """
    new_trajectories = deepcopy(trajectories)
    rigs_recover_inplace(new_trajectories, rigs, master_sensors)
    return new_trajectories


def rigs_recover_inplace(
        trajectories: Trajectories,
        rigs: Rigs,
        master_sensors: Optional[List[str]] = None,
        max_depth: int = 10
):
    """
    Updates the given trajectories by replacing sensor poses by the rig poses,
    each time the sensor is registered in a rig.
    The rig configuration must be given.
    Warning: in some configuration, it may leads to inconsistent results. See code for details.

    :param trajectories: input/output Trajectories.
    :param rigs: input Rigs configuration.
    :param master_sensors: input If given, only compute rig poses for the given sensors.
    :param max_depth: maximum nested rig depth.
    """

    # sensor_id -> rig_id, pose_rig_from_sensor
    reverse_rig_dict = {
        sensor_id: (rig_id, rigs[rig_id, sensor_id].inverse())
        for rig_id, sensor_id in rigs.key_pairs()
    }

    for iteration in range(max_depth):
        # do the replacement while there is sensor_id in trajectories, that can be converted to rig_id
        jobs = [(timestamp, sensor_id, pose_sensor_from_world)
                for timestamp, sensor_id, pose_sensor_from_world in flatten(trajectories, is_sorted=True)
                if sensor_id in reverse_rig_dict]

        if len(jobs) == 0:
            break

        for timestamp, sensor_id, pose_sensor_from_world in jobs:
            # if the sensor is part of a rig, set the pose of the rig,
            # instead of the pose of the sensor
            rig_id, pose_rig_from_sensor = reverse_rig_dict[sensor_id]
            pose_sensor_from_world = trajectories[timestamp].pop(sensor_id)

            # only compute rig poses for master_sensors if any.
            if master_sensors is not None and sensor_id not in master_sensors:
                continue

            # skip if rig pose already recovered.
            if rig_id in trajectories[timestamp]:
                continue

            # warning: if multiple sensors can be used to infer the rig pose
            # (eg. no master sensor, or multiple master sensors), only one sensor is used to compute the rig pose.
            # Since we use is_sorted=True, the selected sensor should be consistent, when present. But if for some
            # timestamp, the usual first sensor is missing, then, another sensor is used as reference and
            # if the sensors dos not actually use the rig calibration (no rigid transform between sensors),
            # it may end up to inconsistent results.
            pose_rig_from_world = PoseTransform.compose([pose_rig_from_sensor, pose_sensor_from_world])
            trajectories[timestamp, rig_id] = pose_rig_from_world


def compute_intermediate_pose(timestamp: int,
                              low_ts: int, low_p: PoseTransform,
                              up_ts: int, up_p: PoseTransform) -> PoseTransform:
    """
    Compute an intermediate pose between two poses.
    It does not come from the recorded data, but is purely computed by interpolation based on given poses.
    We suppose that we move at a regular speed.

    :param timestamp: the timestamp at which time to compute the pose
    :param low_ts: the first timestamp
    :param low_p: the first pose
    :param up_ts: the second timestamp
    :param up_p: the second pose
    :return: the computed pose
    """
    rotation = quaternion.slerp(low_p.r, up_p.r, low_ts, up_ts, timestamp)
    # translation = t0 + (ts-ts0)/(ts1-ts0) * (t1 - t0)
    translation = [low_p.t[0] + (timestamp - low_ts) / (up_ts - low_ts) * (up_p.t[0] - low_p.t[0]),
                   low_p.t[1] + (timestamp - low_ts) / (up_ts - low_ts) * (up_p.t[1] - low_p.t[1]),
                   low_p.t[2] + (timestamp - low_ts) / (up_ts - low_ts) * (up_p.t[2] - low_p.t[2])]
    return PoseTransform(rotation, translation)


def trajectory_transform_inplace(
        trajectories: Trajectories,
        pose_transform_pre: PoseTransform = PoseTransform(),
        pose_transform_post: PoseTransform = PoseTransform()
):
    """
    Apply a PoseTransform to all poses in trajectories.
    new_pose = compose([pose_transform_pre, pose, pose_transform_post])

    :param trajectories: the trajectories to bu updated
    :param pose_transform_pre:
    :param pose_transform_post:
    :return:
    """
    for timestamp, sensor_id, pose in flatten(trajectories):
        trajectories[timestamp, sensor_id] = PoseTransform.compose([pose_transform_pre, pose, pose_transform_post])


def trajectory_rescale_inplace(
        trajectories: Trajectories,
        scale: float
):
    """ apply scale factor to trajectories (translation part only)

    :param trajectories: the trajectories to bu updated.
    :param scale: scale factor
    """
    for timestamp, sensor_id, pose in flatten(trajectories):
        pose.rescale(scale)
