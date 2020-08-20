# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from copy import deepcopy
from typing import Set, Any
import numpy as np


class ImageFeaturesBase(Set[str]):
    """
    base class for per image features (keypoints, descriptors, global_features)
    """

    # features = set(path_to_image_file)
    # features.feature_filenames = set(path_to_feature_file)
    def __init__(self,
                 type_name: str,
                 dtype: Any,
                 dsize: int,
                 *args, **kwargs):
        """

        :param type_name: input name of feature type (eg. SIFT). Used for information only.
        :param dtype: input type of the feature (eg. numpy.float32).
        :param dsize: input length of the feature (eg. 64 for sift descriptor).
        :param args: anything that can be passed to the set() constructor.
        :param kwargs: anything that can be passed to the set() constructor.
        """
        if not isinstance(type_name, str):
            raise TypeError('Expect str as name')
        if not isinstance(dtype, (type, np.dtype)):
            raise TypeError('Expect type (or numpy.dtype) as dtype')
        if not isinstance(dsize, int):
            raise TypeError('Expect int as dsize')
        self._tname = type_name
        self._dtype = dtype
        self._dsize = dsize
        super().__init__(*args, **kwargs)

    @property
    def type_name(self) -> str:
        """
         :return: type name as string
        """
        return self._tname

    @property
    def dtype(self) -> type:
        """
        :return: feature type
        """
        return self._dtype

    @property
    def dsize(self) -> int:
        """
        :return: feature size
        """
        return self._dsize

    def __repr__(self):
        representation = f'{self.type_name} ({self.dtype.__name__} x {self.dsize}) = '
        if len(self) == 0:
            representation += '[]'
        else:
            representation += '[\n' + ',\n'.join(f'\t{i}' for i in self) + '\n]'
        return representation

    def __copy__(self):
        result = type(self)(self.type_name, self.dtype, self.dsize)
        result.update(self)
        return result

    def __deepcopy__(self, memo):
        result = type(self)(self.type_name, self.dtype, self.dsize)
        for v in self:
            result.add(deepcopy(v))
        return result


class Keypoints(ImageFeaturesBase):
    """
    Key points features
    """
    pass


class Descriptors(ImageFeaturesBase):
    """
    Image descriptor features
    """
    pass


class GlobalFeatures(ImageFeaturesBase):
    """
    Global features
    """
    pass
