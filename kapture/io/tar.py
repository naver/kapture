import kapture
import tarfile
import numpy as np
import io
from typing import Any, Dict, List, Type, Union
# import loop ?
from kapture.io.csv import list_features
from kapture.io.features import FEATURES_DATA_DIRNAMES
import os.path as path
import os


logger = kapture.logger


class TarHandler:
    def __init__(self, tarfile_path: str, mode: str = 'r'):
        assert mode in {'r', 'a'}
        self.mode = mode
        self.fid = tarfile.TarFile(tarfile_path, mode=self.mode)
        # list all files
        tarcontent = self.fid.getmembers()
        # if c.name is found multiple time, the value will correspond to its last occurence, so the most up to data one
        self.content = {c.name: c for c in tarcontent}

    def flush(self):
        self.fid.fileobj.flush()

    def close(self):
        self.fid.close()

    def add_array_to_tar(self, filepath: str, data_array: np.ndarray) -> None:
        assert self.mode == 'a'
        info = tarfile.TarInfo(filepath)
        data = data_array.tobytes()
        info.size = len(data)
        self.fid.addfile(tarinfo=info, fileobj=io.BytesIO(data))
        self.flush()
        # self.content[filepath] = info
        self.content[filepath] = self.fid.getmember(filepath)

    def get_array_from_tar(self, filepath: str, dtype: Type, dsize: int) -> np.ndarray:
        assert self.mode == 'r'
        info = self.content[filepath]
        data_array = np.frombuffer(self.fid.extractfile(info).read(), dtype=dtype)
        data_array = data_array.reshape((-1, dsize))
        return data_array


class TarCollection:

    def __init__(self,
                 keypoints: Dict[str, TarHandler] = {},
                 descriptors: Dict[str, TarHandler] = {},
                 global_features: Dict[str, TarHandler] = {},
                 matches: Dict[str, TarHandler] = {}):
        super().__init__()
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.global_features = global_features
        self.matches = matches

    def close(self):
        """
        closes all handlers
        """
        for _, handler in self.keypoints.items():
            handler.close()
        self.keypoints.clear()
        for _, handler in self.descriptors.items():
            handler.close()
        self.descriptors.clear()
        for _, handler in self.global_features.items():
            handler.close()
        self.global_features.clear()
        for _, handler in self.matches.items():
            handler.close()
        self.matches.clear()

    @property
    def keypoints(self) -> Dict[str, TarHandler]:
        """
        :return: the keypoints tar handlers
        """
        return self._keypoints

    @keypoints.setter
    def keypoints(self, keypoints: Dict[str, TarHandler]):
        if not isinstance(keypoints, dict):
            raise TypeError('dict of TarHandler expected')
        self._keypoints = keypoints

    @property
    def descriptors(self) -> Dict[str, TarHandler]:
        """
        :return: the descriptors tar handlers
        """
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors: Dict[str, TarHandler]):
        if not isinstance(descriptors, dict):
            raise TypeError('dict of TarHandler expected')
        self._descriptors = descriptors

    @property
    def global_features(self) -> Dict[str, TarHandler]:
        """
        :return: the global features tar handlers
        """
        return self._global_features

    @global_features.setter
    def global_features(self, global_features: Dict[str, TarHandler]):
        if not isinstance(global_features, dict):
            raise TypeError('dict of TarHandler expected')
        self._global_features = global_features

    @property
    def matches(self) -> Dict[str, TarHandler]:
        """
        :return: the matches tar handlers
        """
        return self._matches

    @matches.setter
    def matches(self, matches: Dict[str, TarHandler]):
        if not isinstance(matches, dict):
            raise TypeError('dict of TarHandler expected')
        self._matches = matches


KAPTURE_TARABLE_TYPES = {
    kapture.Keypoints,
    kapture.Descriptors,
    kapture.GlobalFeatures,
    kapture.Matches
}

FEATURES_TAR_FILENAMES = {
    kapture.Keypoints: lambda x: path.join('reconstruction', 'keypoints', x, 'keypoints.tar'),
    kapture.Descriptors: lambda x: path.join('reconstruction', 'descriptors', x, 'descriptors.tar'),
    kapture.GlobalFeatures: lambda x: path.join('reconstruction', 'global_features', x, 'global_features.tar'),
    kapture.Matches: lambda x: path.join('reconstruction', 'matches', x, 'matches.tar'),
}


def get_feature_tar_fullpath(kapture_type: Any, feature_name: str, kapture_dirpath: str = '') -> str:
    """
    Returns the full path to tar kapture file for a given datastructure and root directory.
    This path is the concatenation of the kapture root path and subpath into kapture into data structure.

    :param kapture_type: type of kapture data (kapture.RecordsCamera, kapture.Trajectories, ...)
    :param kapture_dirpath: root kapture path
    :return: full path of tar file for that type of data
    """
    assert kapture_type in FEATURES_TAR_FILENAMES
    filename = FEATURES_TAR_FILENAMES[kapture_type](feature_name)
    return path.join(kapture_dirpath, filename)


def get_all_tar_handlers(kapture_dir_path: str,
                         mode: str = 'r',
                         skip_list: List[Type[Union[
                             kapture.Keypoints,
                             kapture.Descriptors,
                             kapture.GlobalFeatures,
                             kapture.Matches
                         ]]] = []) -> TarCollection:
    assert mode in {'r', 'a'}
    data_dir_paths = {dtype: path.join(kapture_dir_path, dir_name)
                      for dtype, dir_name in FEATURES_DATA_DIRNAMES.items()}
    kapture_loadable_data = {
        kapture_type
        for kapture_type in KAPTURE_TARABLE_TYPES
        if kapture_type not in skip_list and path.exists(data_dir_paths[kapture_type])
    }
    tar_collection = TarCollection()

    # keypoints
    if kapture.Keypoints in kapture_loadable_data:
        logger.debug(f'opening keypoints tars {data_dir_paths[kapture.Keypoints]} ...')
        keypoints_list = list_features(kapture.Keypoints, kapture_dir_path)
        if len(keypoints_list) > 0:
            for keypoints_type in keypoints_list:
                tarfile_path = get_feature_tar_fullpath(kapture.Keypoints, keypoints_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.keypoints[keypoints_type] = TarHandler(tarfile_path, mode)
    # descriptors
    if kapture.Descriptors in kapture_loadable_data:
        logger.debug(f'opening descriptors tars {data_dir_paths[kapture.Descriptors]} ...')
        descriptors_list = list_features(kapture.Descriptors, kapture_dir_path)
        if len(descriptors_list) > 0:
            for descriptors_type in descriptors_list:
                tarfile_path = get_feature_tar_fullpath(kapture.Descriptors, descriptors_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.descriptors[descriptors_type] = TarHandler(tarfile_path, mode)
    # global_features
    if kapture.GlobalFeatures in kapture_loadable_data:
        logger.debug(f'opening global_features tars {data_dir_paths[kapture.GlobalFeatures]} ...')
        global_features_list = list_features(kapture.GlobalFeatures, kapture_dir_path)
        if len(global_features_list) > 0:
            for global_features_type in global_features_list:
                tarfile_path = get_feature_tar_fullpath(kapture.GlobalFeatures, global_features_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.global_features[global_features_type] = TarHandler(tarfile_path, mode)
    # matches
    if kapture.Matches in kapture_loadable_data:
        logger.debug(f'opening matches tars {data_dir_paths[kapture.Matches]} ...')
        keypoints_list = [name
                          for name in os.listdir(data_dir_paths[kapture.Matches])
                          if os.path.isdir(os.path.join(data_dir_paths[kapture.Matches], name))]
        if len(keypoints_list) > 0:
            for keypoints_type in keypoints_list:
                tarfile_path = get_feature_tar_fullpath(kapture.Matches, keypoints_type, kapture_dir_path)
                if path.isfile(tarfile_path):
                    tar_collection.matches[keypoints_type] = TarHandler(tarfile_path, mode)
    return tar_collection
