# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import numpy as np
from dataclasses import dataclass, astuple, asdict, fields
from typing import Union, Dict, List, Tuple, TypeVar

"""
Records contains sensor recordings.
Each recording (eg. an image) is an entry in Records, with a device ID and a timestamp.
There is to kind of Records :
 - Files: big recorded data (from camera, lidar, depth camera, ...) are stored in separated binary files.
 - Array: small recorded data (from wifi, gnss) are directly stored in kapture text files.

"""


########################################################################################################################
T = TypeVar('T')  # Declare generic type variable


class RecordsBase(Dict[int, Dict[str, T]]):
    """
    brief: Records
            records[timestamp][sensor_id] = <Record>
            or
            records[timestamp, sensor_id] = <Record>
    """

    def __setitem__(self,
                    key: Union[int, Tuple[int, str]],
                    value: Union[Dict[str, T], T]):
        """
        Inserts/changes a record.

        :param key: can be timestamp or (timestamp, sensor_id)
        :param value: can be respectively a dict{sensor_id: DataRecords} or DataRecords
        :return:
        """
        # enforce type checking
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            if not isinstance(value, self.record_type):
                raise TypeError(f'invalid record type of {type(value)} (expect {self.record_type})')
            self.setdefault(timestamp, {})[device_id] = value
        elif isinstance(key, int):
            # key is a timestamp
            timestamp = key
            if not isinstance(value, dict):
                raise TypeError('invalid value for data (expect dict)')
            if not all(isinstance(k, str) for k in value.keys()):
                raise TypeError('invalid device_id')
            if not all(isinstance(k, self.record_type) for k in value.values()):
                raise TypeError(f'invalid value for record (expect all {self.record_type})')
            super(RecordsBase, self).__setitem__(timestamp, value)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __getitem__(self, key: Union[int, Tuple[int, str]]) -> Union[Dict[str, T], T]:
        """
        Returns a single record for a (timestamp, sensor) or all records for a timestamp.

        :param key: can be timestamp or (timestamp, sensor_id)
        :return: return respectively a dict{sensor_id: DataRecords} or DataRecords
        """
        if isinstance(key, tuple):
            # key is a pair of (timestamp, device_id)
            timestamp = key[0]
            device_id = key[1]
            if not isinstance(timestamp, int):
                raise TypeError('invalid timestamp')
            if not isinstance(device_id, str):
                raise TypeError('invalid device_id')
            return super(RecordsBase, self).__getitem__(timestamp)[device_id]
        elif isinstance(key, int):
            # key is a timestamp
            return super(RecordsBase, self).__getitem__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def key_pairs(self) -> List[Tuple[int, str]]:
        """
        Returns a list of (timestamp, device_id) contained in records.
        Those pairs can be used to access a single record data.

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
            return super(RecordsBase, self).__contains__(timestamp) and device_id in self[timestamp]
        elif isinstance(key, int):
            return super(RecordsBase, self).__contains__(key)
        else:
            raise TypeError('key must be Union[int, Tuple[int, str]]')

    def __repr__(self) -> str:
        # [timestamp, sensor_id] = str
        lines = [f'[ {timestamp:010}, {sensor_id:5}] = {data}'
                 for timestamp, sensors in self.items()
                 for sensor_id, data in sensors.items()]
        return '\n'.join(lines)


class RecordsFilePath(RecordsBase[str]):
    """
    Brief: base class for records pointing to a data file (eg. camera, lidar).
    """
    record_type = str

    def __setitem__(self,
                    key: Union[int, Tuple[int, str]],
                    value: Union[Dict[str, str], str]):
        """ see RecordsBase.__setitem__ """
        if isinstance(key, tuple):
            if not isinstance(value, str):
                raise TypeError('invalid data')
        elif isinstance(key, int):
            if not isinstance(value, dict):
                raise TypeError('invalid value for data')
            if not all(isinstance(v, str) for v in value.values()):
                raise TypeError('invalid data')
        super(RecordsFilePath, self).__setitem__(key, value)


class RecordsCamera(RecordsFilePath):
    """
    Camera records
    """
    pass


class RecordsDepth(RecordsFilePath):
    """
    Depth map records
    """
    dtype = np.float32


class RecordsLidar(RecordsFilePath):
    """
    Lidar records
    """
    pass


# Record Array #########################################################################################################
# @dataclass
class RecordArray:
    def __post_init__(self):
        # force cast to expected types
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, field.type):
                setattr(self, field.name, field.type(value))

    def astuple(self):
        return astuple(self)

    def asdict(self):
        return asdict(self)

    @classmethod
    def fields(cls):
        return fields(cls)


class RecordsArray(RecordsBase[T]):
    """
    brief: Records
            records[timestamp][sensor_id] = RecordArray
            or
            records[timestamp, sensor_id] = RecordArray
    """
    pass


# wifi recordings is made of dict of signals (with signal strength).
@dataclass
class RecordWifiSignal(RecordArray):
    frequency: int
    rssi: float
    ssid: str = ''
    scan_time_start: int = 0
    scan_time_end: int = 0


class RecordWifi(dict):
    def __setitem__(self, bssid: str, data: RecordWifiSignal):
        if not isinstance(bssid, str):
            raise TypeError(f'{bssid} is not expected type str.')
        if not isinstance(data, RecordWifiSignal):
            raise TypeError(f'{data} is not expected type RecordWifiHotspot.')
        super().__setitem__(bssid, data)


class RecordsWifi(RecordsArray[RecordWifi]):
    """
    brief: Records wifi
            records[timestamp][sensor_id] =  <RecordWifi> = {bssid: RecordWifiHotspot}
            or
            records[timestamp, sensor_id] = <RecordWifi>
    """
    record_type = RecordWifi


# bluetooth recordings is made of dict of bt devices fingerprints (with signal strength).
@dataclass
class RecordBluetoothSignal(RecordArray):
    rssi: float
    name: str = ''


class RecordBluetooth(dict):
    def __setitem__(self, address: str, data: RecordBluetoothSignal):
        if not isinstance(address, str):
            raise TypeError(f'{address} is not expected type str.')
        if not isinstance(data, RecordBluetoothSignal):
            raise TypeError(f'{data} is not expected type RecordBluetoothDevice.')
        super().__setitem__(address, data)


class RecordsBluetooth(RecordsArray[RecordBluetooth]):
    """
    brief: Records wifi
            records[timestamp][sensor_id] = <RecordBluetooth> = {address: <RecordBluetoothDevice>}
            or
            records[timestamp, sensor_id] = <RecordBluetooth>
    """
    record_type = RecordBluetooth


# gnss recordings
@dataclass
class RecordGnss(RecordArray):
    x: float
    y: float
    z: float
    utc: int
    dop: float = 0.


class RecordsGnss(RecordsArray[RecordGnss]):
    record_type = RecordGnss


# Accelerometer recordings
@dataclass
class RecordAccelerometer(RecordArray):
    x_accel: float
    y_accel: float
    z_accel: float


class RecordsAccelerometer(RecordsArray[RecordAccelerometer]):
    record_type = RecordAccelerometer


# Gyroscope recordings
@dataclass
class RecordGyroscope(RecordArray):
    x_speed: float
    y_speed: float
    z_speed: float


class RecordsGyroscope(RecordsArray[RecordGyroscope]):
    record_type = RecordGyroscope


# Magnetic field recordings
@dataclass
class RecordMagnetic(RecordArray):
    x_strength: float
    y_strength: float
    z_strength: float


class RecordsMagnetic(RecordsArray[RecordMagnetic]):
    record_type = RecordMagnetic
