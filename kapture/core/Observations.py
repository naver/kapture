# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from typing import Dict, List, Tuple


class Observations(Dict[int, List[Tuple[str, int]]]):
    """
    Observations. This can be used like this:

    - observations[point3d_idx]= list( observation )
    - observation = (image_path, keypoint_idx)
    """

    def add(self, point3d_idx: int, image_filename: str, keypoint_idx: int):
        """
        Adds a 2-D observation (image, keypoint) of a 3D point.

        :param point3d_idx: index of the 3D point to add an observation of.
        :param image_filename: name of the image where the 3D points is observed
        :param keypoint_idx: index of the keypoints in the image that correspond to the 3D point.
        :return:
        """
        # enforce type checking
        if not isinstance(point3d_idx, int):
            raise TypeError('invalid type for point3d_idx')
        if not isinstance(image_filename, str):
            raise TypeError('invalid type for image_filename')
        if not isinstance(keypoint_idx, int):
            raise TypeError('invalid type for keypoint_idx')
        self.setdefault(point3d_idx, []).append((image_filename, keypoint_idx))

    def __repr__(self):
        representation = ''
        for point3d_idx in sorted(self.keys()):
            representation += f'[{point3d_idx:05}]: '
            for image_path, keypoint_idx in self.get(point3d_idx):
                representation += f'\t({image_path}, {keypoint_idx})'
            representation += '\n'
        return representation
