# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from typing import Dict, List, Tuple, Union


class Observations(Dict[int, Dict[str, List[Tuple[str, int]]]]):
    """
    Observations. This can be used like this:

    - observations[point3d_idx][keypoints_type] = list( observation )
    - observation = (image_path, keypoint_idx)
    """

    def add(self, point3d_idx: int, keypoints_type: str, image_filename: str, keypoint_idx: int):
        """
        Adds a 2-D observation (image, keypoint) of a 3D point.

        :param point3d_idx: index of the 3D point to add an observation of.
        :param keypoints_type: type of keypoints, name of the keypoints subfolder
        :param image_filename: name of the image where the 3D points is observed
        :param keypoint_idx: index of the keypoints in the image that correspond to the 3D point.
        :return:
        """
        # enforce type checking
        if not isinstance(point3d_idx, int):
            raise TypeError('invalid type for point3d_idx')
        if not isinstance(keypoints_type, str):
            raise TypeError('invalid type for keypoints_type')
        if not isinstance(image_filename, str):
            raise TypeError('invalid type for image_filename')
        if not isinstance(keypoint_idx, int):
            raise TypeError('invalid type for keypoint_idx')
        self.setdefault(point3d_idx, {}).setdefault(keypoints_type, []).append((image_filename, keypoint_idx))

    def __getitem__(self, key: Union[int, Tuple[int, str]]) -> Union[Dict[str, List[Tuple[str, int]]],
                                                                     List[Tuple[str, int]]]:
        if isinstance(key, tuple):
            # key is a pair of (point3d_idx, keypoints_type)
            point3d_idx = key[0]
            keypoints_type = key[1]
            if not isinstance(point3d_idx, int):
                raise TypeError('invalid point3d_idx')
            if not isinstance(keypoints_type, str):
                raise TypeError('invalid keypoints_type')
            return super(Observations, self).__getitem__(point3d_idx)[keypoints_type]
        elif isinstance(key, int):
            # key is a point3d_idx
            return super(Observations, self).__getitem__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def key_pairs(self) -> List[Tuple[int, str]]:
        """
        Returns the list of (point3d_idx, keypoints_type) contained in observations.
        Those pairs can be used to access a list of observation.
        :return: list of (point3d_idx, keypoints_type)
        """
        return [
            (point3d_idx, keypoints_type)
            for point3d_idx, per_feature_observations in self.items()
            for keypoints_type in per_feature_observations.keys()
        ]

    def observations_number(self) -> int:
        """
        Get the number of observations
        """
        nb = 0
        for per_feature_observations in self.values():
            for observations_list in per_feature_observations.values():
                nb += len(observations_list)
        return nb

    def __contains__(self, key: Union[int, Tuple[int, str]]):
        if isinstance(key, tuple):
            # key is a pair of (point3d_idx, keypoints_type)
            point3d_idx = key[0]
            keypoints_type = key[1]
            if not isinstance(point3d_idx, int):
                raise TypeError('invalid point3d_idx')
            if not isinstance(keypoints_type, str):
                raise TypeError('invalid keypoints_type')
            return super(Observations, self).__contains__(point3d_idx) and keypoints_type in self[point3d_idx]
        elif isinstance(key, int):
            return super(Observations, self).__contains__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __repr__(self) -> str:
        representation = ''
        # [point3d_idx, keypoints_type]:   (image_path, keypoint_idx)   (image_path, keypoint_idx)...
        for point3d_idx, keypoints_type in sorted(self.key_pairs(), key=lambda x: x[0]):
            representation += f'[{point3d_idx:05}, {keypoints_type}]: '
            assert point3d_idx is not None
            for image_path, keypoint_idx in self.get(point3d_idx)[keypoints_type]:
                representation += f'\t({image_path}, {keypoint_idx})'
            representation += '\n'
        return representation
