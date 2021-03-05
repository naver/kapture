import tarfile
import numpy as np
import io
from typing import Type


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
