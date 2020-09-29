# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from .PoseTransform import PoseTransform
from .Rigs import Rigs
from .flatten import flatten
from typing import Union, Dict, List, Tuple, Optional
from copy import deepcopy


class Trajectories(Dict[int, Dict[str, PoseTransform]]):
    """
    brief: Trajectories
            trajectories[timestamp][sensor_id] = <PoseTransform>
            or
            trajectories[(timestamp, sensor_id)] = <PoseTransform>

            with <PoseTransform> transforming points from world to device.
    """

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

    def __contains__(self, key: Union[int, Tuple[int, str]]):
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            return super(Trajectories, self).__contains__(timestamp) and device_id in self[timestamp]
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
    :return:
    """
    assert isinstance(rigs, Rigs)
    assert isinstance(trajectories, Trajectories)
    # collect all poses of rigs in trajectories
    for iteration in range(max_depth):
        # repeat the operation while there is so rig remaining (nested rigs)
        jobs = [(timestamp, rig_id, pose_rig_from_world)
                for timestamp, rig_id, pose_rig_from_world in flatten(trajectories)
                if rig_id in rigs.keys()]

        if len(jobs) == 0:
            # we are done
            break

        # replace those rigs poses by the one of the sensors
        for timestamp, rig_id, pose_rig_from_world in jobs:
            # its a rig, add every sensors in it instead.
            for device_id, pose_device_from_rig in rigs[rig_id].items():
                pose_cam_from_world = PoseTransform.compose([pose_device_from_rig, pose_rig_from_world])
                trajectories[timestamp, device_id] = pose_cam_from_world
            # then remove this rig pose
            del trajectories[timestamp][rig_id]

    # remove useless (empty) timestamp (if any) from trajectories
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
    :return:
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
