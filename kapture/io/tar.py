from kapture.utils.paths import path_secure
import kapture
import tarfile
import numpy as np
import io
from typing import Any, Dict, Iterable, List, Optional, Type, Union
import os.path as path


logger = kapture.logger


class TarHandler:
    def __init__(self, tarfile_path: str, mode: str = 'r'):
        assert mode in {'r', 'a'}
        self.mode = mode
        self.fid = tarfile.TarFile(tarfile_path, mode=self.mode)
        # list all files
        tarcontent = self.fid.getmembers()
        # if c.name is found multiple time, the value will correspond to its last occurence, so the most up to data one
        self.content = {path_secure(c.name): c for c in tarcontent}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def flush(self):
        self.fid.fileobj.flush()

    def close(self):
        self.fid.close()

    def add_array_to_tar(self, filepath: str, data_array: np.ndarray) -> None:
        assert self.mode == 'a'
        filepath = path_secure(filepath)
        info = tarfile.TarInfo(filepath)
        data = data_array.tobytes()
        info.size = len(data)
        self.fid.addfile(tarinfo=info, fileobj=io.BytesIO(data))
        self.flush()
        # self.content[filepath] = info
        self.content[filepath] = self.fid.getmember(filepath)

    def get_array_from_tar(self, filepath: str, dtype: Type, dsize: int) -> np.ndarray:
        assert self.mode == 'r'
        filepath = path_secure(filepath)
        if filepath not in self.content:
            print(self.content)
        info = self.content[filepath]
        data_array = np.frombuffer(self.fid.extractfile(info).read(), dtype=dtype)
        data_array = data_array.reshape((-1, dsize))
        return data_array


class TarCollection:

    def __init__(self,
                 keypoints: Optional[Dict[str, TarHandler]] = None,
                 descriptors: Optional[Dict[str, TarHandler]] = None,
                 global_features: Optional[Dict[str, TarHandler]] = None,
                 matches: Optional[Dict[str, TarHandler]] = None):
        super().__init__()
        if keypoints is None:
            self.keypoints = {}
        else:
            self.keypoints = keypoints

        if descriptors is None:
            self.descriptors = {}
        else:
            self.descriptors = descriptors

        if global_features is None:
            self.global_features = {}
        else:
            self.global_features = global_features

        if matches is None:
            self.matches = {}
        else:
            self.matches = matches

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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


def list_files_in_tar(tar_handler: TarHandler,
                      filename_extensions: Optional[Union[str, List[str]]] = None) -> Iterable[str]:
    """
    Returns the list of file path into the given tar.
    If a list of extensions is given, returns only files with those extensions.

    :param tar_handler: opened tar reference
    :param filename_extensions: optional file name extensions to filter in
    :return: list of paths
    """
    # list all files
    file_paths = tar_handler.content.keys()

    # if extensions are given, keep only files that complies.
    if filename_extensions:
        if not isinstance(filename_extensions, list):
            # make sure extensions is a list
            filename_extensions = [filename_extensions]
        # make sure given extensions are lower case.
        filename_extensions = [ext.lower() for ext in filename_extensions]
        # check the extension is authorized
        file_paths = (
            file_path for file_path in file_paths
            if path.splitext(file_path)[1].lower() in filename_extensions
        )

    file_paths = (path_secure(file_path)
                  for file_path in file_paths)
    return file_paths


def retrieve_tar_handler_from_collection(kapture_type: Type[Union[kapture.Keypoints,
                                                                  kapture.Descriptors,
                                                                  kapture.GlobalFeatures,
                                                                  kapture.Matches]],
                                         feature_type: str,
                                         tar_handlers: Optional[Union[TarCollection,
                                                                      TarHandler]] = None) -> Optional[TarHandler]:
    """
    get a tar_handler from a collection of tar_handlers. the result will be None if it's not in a Tar

    :param kapture_type: kapture class type.
    :param feature_type: the name of the features type
    :param tar_handler: None or collection of preloaded tar archives, defaults to None
    :return: a TarHandler if the combo kapture_type/feature_type is in a tar, or None if it's in a directory
    """
    tar_local_handler = None
    if tar_handlers is not None:
        if isinstance(tar_handlers, TarCollection):
            if kapture_type == kapture.Keypoints and feature_type in tar_handlers.keypoints:
                tar_local_handler = tar_handlers.keypoints[feature_type]
            elif kapture_type == kapture.Descriptors and feature_type in tar_handlers.descriptors:
                tar_local_handler = tar_handlers.descriptors[feature_type]
            elif kapture_type == kapture.GlobalFeatures and feature_type in tar_handlers.global_features:
                tar_local_handler = tar_handlers.global_features[feature_type]
            elif kapture_type == kapture.Matches and feature_type in tar_handlers.matches:
                tar_local_handler = tar_handlers.matches[feature_type]
        elif isinstance(tar_handlers, TarHandler):
            tar_local_handler = tar_handlers
        else:
            raise TypeError(f'unknown {tar_handlers}')
    return tar_local_handler
