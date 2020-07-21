# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from .PoseTransform import PoseTransform
from typing import Dict, Union, Tuple, List


class Rigs(Dict[str, Dict[str, PoseTransform]]):
    """
    brief: Rigs is a set of rig. Each rig is uniquely defined by a sensor_id.
        rigs[rig_id][sensor_id] = <PoseTransform>
        or
        rigs[rig_id, sensor_id] = <PoseTransform>

        with <PoseTransform> transforming points from rig to camera.
    """

    def __setitem__(self,
                    key: Union[str, Tuple[str, str]],
                    value: Union[Dict[str, PoseTransform], PoseTransform]):
        # enforce type checking
        if isinstance(key, tuple):
            # rigs[rig_id, sensor_id] = <PoseTransform>
            assert len(key) == 2
            if any(not isinstance(x, str) for x in key):
                raise TypeError('expect str type as key pair')
            rig_device_id, sensor_id = key[0], key[1]
            if not isinstance(value, PoseTransform):
                raise TypeError('expect PoseTransform type as value')
            self.setdefault(rig_device_id, {})[sensor_id] = value

        elif isinstance(key, str):
            # key is rig_id
            rig_id = key
            # rigs[rig_id] = {}
            # check type of values?
            if not isinstance(value, dict):
                raise TypeError('invalid value for rig id.')
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError('invalid device_id')
            if not all(isinstance(v, PoseTransform) for v in value.values()):
                raise TypeError('invalid Pose')
            super(Rigs, self).__setitem__(rig_id, value)
        else:
            raise TypeError('invalid key type for Rigs')

    def __getitem__(self, key: Union[str, Tuple[str, str]]) -> Union[Dict[str, PoseTransform], PoseTransform]:
        if isinstance(key, str):
            # rigs[rig_id] = <Rig>
            return super(Rigs, self).__getitem__(key)
        elif isinstance(key, tuple):
            assert len(key) == 2
            rig_id, sensor_id = key
            if not isinstance(rig_id, str):
                raise TypeError('invalid rig_id')
            if not isinstance(sensor_id, str):
                raise TypeError('invalid sensor_id')
            return super(Rigs, self).__getitem__(rig_id)[sensor_id]
        else:
            raise TypeError('key must be either str or Tuple[str, str]')

    def key_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns the list of (rig_id, device_id) contained in trajectories.
        Those pairs can be used to access a single trajectory pose.

        :return: list of (rig_id, device_id)
        """
        return [
            (rig_id, sensor_id)
            for rig_id, sensors in self.items()
            for sensor_id in sensors.keys()
        ]

    def __repr__(self) -> str:
        # [rig_id, sensor_id] = qw, qx, qy, qz, tx, ty, tz
        poses = [f'[{rig_id:5}, {sensor_id:5}] = {pose}'
                 for rig_id, rig in self.items()
                 for sensor_id, pose in rig.items()]
        return '\n'.join(poses)
