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

    def __init__(self,
                 type_name: str,
                 dtype: Any,
                 dsize: int,
                 keypoints_type: str,
                 metric_type: str,
                 *args, **kwargs):
        """


        :param type_name: input name of feature type (eg. SIFT). Used for information only.
        :param dtype: input type of the feature (eg. numpy.float32).
        :param dsize: input length of the feature (eg. 64 for sift descriptor).
        :param keypoints_type: type of keypoints, name of the keypoints subfolder
        :param metric_type: name of the metric used to compare the descriptors, for information only.
        :param args: anything that can be passed to the set() constructor.
        :param kwargs: anything that can be passed to the set() constructor.
        """
        super().__init__(type_name, dtype, dsize, *args, **kwargs)
        if not isinstance(keypoints_type, str):
            raise TypeError('Expect str as keypoints_type')
        if not isinstance(metric_type, str):
            raise TypeError('Expect str as metric_type')
        self._keypoints_type = keypoints_type
        self._metric_type = metric_type

    @property
    def keypoints_type(self) -> str:
        """
        :return: keypoints_type
        """
        return self._keypoints_type

    @property
    def metric_type(self) -> str:
        """
        :return: metric_type
        """
        return self._metric_type

    def __repr__(self):
        representation = f'{self.type_name} {self.keypoints_type} {self.metric_type}' \
                         f' ({self.dtype.__name__} x {self.dsize}) = '
        if len(self) == 0:
            representation += '[]'
        else:
            representation += '[\n' + ',\n'.join(f'\t{i}' for i in self) + '\n]'
        return representation

    def __copy__(self):
        result = type(self)(self.type_name, self.dtype, self.dsize, self.keypoints_type, self.metric_type)
        result.update(self)
        return result

    def __deepcopy__(self, memo):
        result = type(self)(self.type_name, self.dtype, self.dsize, self.keypoints_type, self.metric_type)
        for v in self:
            result.add(deepcopy(v))
        return result


class GlobalFeatures(ImageFeaturesBase):
    """
    Global features
    """

    def __init__(self,
                 type_name: str,
                 dtype: Any,
                 dsize: int,
                 metric_type: str,
                 *args, **kwargs):
        """


        :param type_name: input name of feature type (eg. SIFT). Used for information only.
        :param dtype: input type of the feature (eg. numpy.float32).
        :param dsize: input length of the feature (eg. 64 for sift descriptor).
        :param metric_type: name of the metric used to compare the descriptors, for information only.
        :param args: anything that can be passed to the set() constructor.
        :param kwargs: anything that can be passed to the set() constructor.
        """
        super().__init__(type_name, dtype, dsize, *args, **kwargs)
        if not isinstance(metric_type, str):
            raise TypeError('Expect str as metric_type')
        self._metric_type = metric_type

    @property
    def metric_type(self) -> str:
        """
        :return: metric_type
        """
        return self._metric_type

    def __repr__(self):
        representation = f'{self.type_name} {self.metric_type} ({self.dtype.__name__} x {self.dsize}) = '
        if len(self) == 0:
            representation += '[]'
        else:
            representation += '[\n' + ',\n'.join(f'\t{i}' for i in self) + '\n]'
        return representation

    def __copy__(self):
        result = type(self)(self.type_name, self.dtype, self.dsize, self.metric_type)
        result.update(self)
        return result

    def __deepcopy__(self, memo):
        result = type(self)(self.type_name, self.dtype, self.dsize, self.metric_type)
        for v in self:
            result.add(deepcopy(v))
        return result
