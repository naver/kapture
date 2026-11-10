# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
Extraction of a part of a kapture: the records and poses of some periods of time,
or of the periods the trajectory spends in an area.
"""

import json
import logging
import os
import os.path as path
import shutil
from bisect import bisect_right
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import kapture
from kapture.io.binary import TransferAction, transfer_files_from_dir
from kapture.io.csv import kapture_from_dir, kapture_to_dir
from kapture.io.records import RECORD_DATA_DIRNAME, get_record_fullpath, import_record_data_from_dir_link_dir
from kapture.io.structure import delete_existing_kapture_files

logger = logging.getLogger('extract')

TIMED_DATA_NAMES = ['trajectories', 'records_camera', 'records_depth', 'records_range', 'records_lidar',
                    'records_wifi', 'records_bluetooth', 'records_gnss', 'records_accelerometer',
                    'records_gyroscope', 'records_magnetic']
""" members of a Kapture indexed by [timestamp][device_id] """

FILE_RECORDS_NAMES = ['records_camera', 'records_depth', 'records_range', 'records_lidar']
""" records whose value is a file of sensors/records_data """

NOT_EXTRACTED_TYPES = [kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures, kapture.Matches,
                       kapture.Points3d, kapture.Observations]
""" reconstruction data, without timestamps: not handled by the extraction of a period """

RAW_DIRNAME = path.join('sensors', 'raw')
EXTRACT_FILENAME = 'subset.json'

Window = Tuple[int, int]
Area = Callable[[np.ndarray], np.ndarray]
""" N x 3 points of the world frame -> N booleans, True inside the area """


def _timed_data(kapture_data: kapture.Kapture) -> Dict[str, dict]:
    """ {member name: trajectories or records} of the timed data present in the kapture """
    timed_data = {name: getattr(kapture_data, name) for name in TIMED_DATA_NAMES}
    return {name: data for name, data in timed_data.items() if data is not None}


def time_range(kapture_data: kapture.Kapture) -> Optional[Window]:
    """
    :return: first and last timestamps of the records and trajectories, None if there is none.
    """
    timestamps = [t for data in _timed_data(kapture_data).values() if data for t in (min(data), max(data))]
    return (min(timestamps), max(timestamps)) if timestamps else None


def normalize_windows(windows: Iterable[Sequence[float]],
                      relative: bool = False,
                      time_unit: float = 1e6,
                      kapture_data: Optional[kapture.Kapture] = None) -> List[Window]:
    """
    Converts periods of time to sorted, disjoint windows of kapture timestamps.

    :param windows: periods (start, end), both included.
    :param relative: the periods are in seconds since the first timestamp of kapture_data,
                     instead of kapture timestamps.
    :param time_unit: number of kapture timestamp units per second (relative periods only).
    :param kapture_data: the kapture the relative periods refer to.
    :return: windows (start, end) of kapture timestamps, sorted, the overlapping ones merged.
    """
    windows = [tuple(window) for window in windows]
    if not windows:
        raise ValueError('no period given')
    if any(len(window) != 2 for window in windows):
        raise ValueError('a period is a pair (start, end)')
    if relative:
        if kapture_data is None or time_range(kapture_data) is None:
            raise ValueError('relative periods need a kapture with timestamps')
        t_min = time_range(kapture_data)[0]
        windows = [(t_min + int(round(t0 * time_unit)), t_min + int(round(t1 * time_unit))) for t0, t1 in windows]
    else:
        windows = [(int(t0), int(t1)) for t0, t1 in windows]
    for t0, t1 in windows:
        if t1 < t0:
            raise ValueError(f'period ends before it starts: {t0} .. {t1}')
    merged = []
    for t0, t1 in sorted(windows):
        if merged and t0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t1))
        else:
            merged.append((t0, t1))
    return merged


def extract_period(kapture_data: kapture.Kapture,
                   windows: List[Window],
                   skip_sensor_ids: Iterable[str] = ()) -> kapture.Kapture:
    """
    Keeps the poses and the records of the given periods. In memory only, the input is left untouched.
    The reconstruction data (features, matches, points3d, observations) are not extracted.

    :param kapture_data: input kapture.
    :param windows: sorted disjoint windows (start, end) of kapture timestamps, both included,
                    see normalize_windows.
    :param skip_sensor_ids: sensors (or rigs) left out.
    :return: the extracted kapture.
    """
    skip_sensor_ids = set(skip_sensor_ids)
    starts = [t0 for t0, _ in windows]

    def in_windows(timestamp: int) -> bool:
        index = bisect_right(starts, timestamp) - 1
        return index >= 0 and timestamp <= windows[index][1]

    extracted = kapture.Kapture()
    extracted.__version__ = kapture_data.format_version
    if kapture_data.sensors is not None:
        extracted.sensors = kapture.Sensors()
        for sensor_id, sensor in kapture_data.sensors.items():
            if sensor_id not in skip_sensor_ids:
                extracted.sensors[sensor_id] = sensor
    if kapture_data.rigs is not None:
        extracted.rigs = kapture.Rigs()
        for rig_id, sensor_id in kapture_data.rigs.key_pairs():
            if rig_id not in skip_sensor_ids and sensor_id not in skip_sensor_ids:
                extracted.rigs[rig_id, sensor_id] = kapture_data.rigs[rig_id, sensor_id]
    for name, data in _timed_data(kapture_data).items():
        kept = type(data)()
        for timestamp in sorted(data):
            if not in_windows(timestamp):
                continue
            of_sensors = {sensor_id: value for sensor_id, value in data[timestamp].items()
                          if sensor_id not in skip_sensor_ids}
            if of_sensors:
                kept[timestamp] = of_sensors
        setattr(extracted, name, kept)
    return extracted


def trajectory_positions(trajectories: kapture.Trajectories,
                         skip_sensor_ids: Iterable[str] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positions of the devices (rigs or sensors) of a trajectory, in the world frame.

    :param trajectories: poses, device from world as kapture stores them.
    :param skip_sensor_ids: devices to ignore.
    :return: timestamps (N, int64, sorted) and positions (N x 3), one per pose.
    """
    skip_sensor_ids = set(skip_sensor_ids)
    timestamps, positions = [], []
    for timestamp in sorted(trajectories):
        for device_id, pose in trajectories[timestamp].items():
            if device_id in skip_sensor_ids or pose.r is None or pose.t is None:
                continue
            timestamps.append(timestamp)
            positions.append(pose.inverse().t.flatten())
    return np.array(timestamps, dtype=np.int64), np.array(positions, dtype=float).reshape(-1, 3)


def box_bounds(box: Sequence[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param box: (xmin, xmax, ymin, ymax, zmin, zmax), box aligned with the axes; None for no bound on that side.
    :return: lower and upper corners (3 values each), -inf / +inf where there is no bound.
    """
    if len(box) != 6:
        raise ValueError('a box is (xmin, xmax, ymin, ymax, zmin, zmax)')
    lower = np.array([-np.inf if value is None else float(value) for value in box[0::2]])
    upper = np.array([np.inf if value is None else float(value) for value in box[1::2]])
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or not np.all(lower < upper):
        raise ValueError(f'empty box: min {lower.tolist()}, max {upper.tolist()}')
    return lower, upper


def points_in_box(points: np.ndarray, box: Sequence[Optional[float]]) -> np.ndarray:
    """
    :param points: N x 3 points.
    :param box: (xmin, xmax, ymin, ymax, zmin, zmax), box aligned with the axes; None for no bound on that side.
    :return: N booleans, True for the points inside (border included).
    """
    lower, upper = box_bounds(box)
    points = np.asarray(points, dtype=float)[:, :3]
    return np.all((lower <= points) & (points <= upper), axis=1)


def points_in_sphere(points: np.ndarray, center: Sequence[float], radius: float) -> np.ndarray:
    """
    :param points: N x 3 points.
    :param center: center (x, y, z) of the sphere.
    :param radius: radius of the sphere.
    :return: N booleans, True for the points inside (border included).
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError('the center of a sphere is (x, y, z)')
    if not radius > 0:
        raise ValueError(f'the radius of a sphere is positive, got {radius}')
    points = np.asarray(points, dtype=float)[:, :3]
    return np.sum((points - center) ** 2, axis=1) <= radius ** 2


def box_area(box: Sequence[Optional[float]]) -> Area:
    """ area = the points inside the box (xmin, xmax, ymin, ymax, zmin, zmax), None for no bound on that side """
    box_bounds(box)  # checks the box right now
    return lambda points: points_in_box(points, box)


def sphere_area(center: Sequence[float], radius: float) -> Area:
    """ area = the points at most radius away from the center """
    points_in_sphere(np.zeros((0, 3)), center, radius)  # checks the sphere right now
    return lambda points: points_in_sphere(points, center, radius)


def windows_in_area(kapture_data: kapture.Kapture,
                    area: Area,
                    gap: float = 5.,
                    pad: float = 0.2,
                    time_unit: float = 1e6,
                    skip_sensor_ids: Iterable[str] = ()) -> List[Window]:
    """
    Periods of time the trajectory spends in an area: the poses whose position is inside the area,
    grouped in periods.

    :param kapture_data: kapture with trajectories.
    :param area: tells which points of the world frame are in the area, see box_area and sphere_area.
    :param gap: two poses inside the area less than gap seconds apart are in the same period
                (a short way out of the area does not cut the period).
    :param pad: seconds added before and after every period.
    :param time_unit: number of kapture timestamp units per second.
    :param skip_sensor_ids: trajectory devices to ignore.
    :return: sorted disjoint windows (start, end) of kapture timestamps, empty if no pose is in the area.
    """
    if not kapture_data.trajectories:
        raise ValueError('the extraction of an area needs trajectories')
    timestamps, positions = trajectory_positions(kapture_data.trajectories, skip_sensor_ids)
    inside = timestamps[area(positions)] if len(timestamps) else timestamps
    if len(inside) == 0:
        return []
    cuts = np.flatnonzero(np.diff(inside) > gap * time_unit)
    starts = inside[np.concatenate([[0], cuts + 1])]
    ends = inside[np.concatenate([cuts, [len(inside) - 1]])]
    pad = int(round(pad * time_unit))
    return normalize_windows([(int(t0) - pad, int(t1) + pad) for t0, t1 in zip(starts, ends)])


def _drop_records_without_file(kapture_data: kapture.Kapture, source_record_dirpath: str) -> Dict[str, int]:
    """ removes the records whose data file is missing in the source, returns their number per sensor """
    missing = {}
    for name in FILE_RECORDS_NAMES:
        records = getattr(kapture_data, name)
        if records is None:
            continue
        for timestamp, sensor_id, filename in list(kapture.flatten(records)):
            if not path.exists(path.join(source_record_dirpath, filename)):
                del records[timestamp, sensor_id]
                missing[sensor_id] = missing.get(sensor_id, 0) + 1
    return missing


def _transfer_raw(source_dirpath: str, destination_dirpath: str, transfer: TransferAction) -> None:
    """ sensors/raw (original files kept by the importers): every entry copied or linked as is """
    if transfer == TransferAction.skip or not path.isdir(source_dirpath):
        return
    if path.lexists(destination_dirpath):
        if path.islink(destination_dirpath):
            os.remove(destination_dirpath)
        else:
            shutil.rmtree(destination_dirpath)
    if transfer == TransferAction.root_link:
        os.symlink(path.realpath(source_dirpath), destination_dirpath)
        return
    os.makedirs(destination_dirpath)
    for entry in sorted(os.listdir(source_dirpath)):
        source = path.realpath(path.join(source_dirpath, entry))
        destination = path.join(destination_dirpath, entry)
        if transfer == TransferAction.copy:
            if path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        elif transfer == TransferAction.link_relative:
            os.symlink(path.relpath(source, path.realpath(destination_dirpath)), destination)
        else:
            os.symlink(source, destination)


def _check_paths(input_path: str, output_path: str, transfer: TransferAction) -> None:
    if transfer == TransferAction.move:
        raise ValueError('transfer "move" would remove the data files from the input kapture')
    if path.exists(output_path) and path.samefile(input_path, output_path):
        raise ValueError('input and output are the same directory')


def _load(input_path: str, skip_sensor_ids: List[str], with_points3d: bool = False) -> kapture.Kapture:
    """ loads the input without the reconstruction data (but points3d on request), warns about what is left out """
    logger.info(f'loading {input_path}')
    skip_list = [data_type for data_type in NOT_EXTRACTED_TYPES
                 if not (with_points3d and data_type is kapture.Points3d)]
    kapture_data = kapture_from_dir(input_path, skip_list=skip_list)
    left_out = [name for name in ['points3d.txt', 'observations.txt', 'keypoints', 'descriptors', 'global_features',
                                  'matches']
                if path.exists(path.join(input_path, 'reconstruction', name))
                and not (with_points3d and name == 'points3d.txt')]
    if left_out:
        logger.warning(f'reconstruction data of the input that are not extracted: {left_out}')
    unknown = [sensor_id for sensor_id in skip_sensor_ids
               if sensor_id not in (kapture_data.sensors or {}) and sensor_id not in (kapture_data.rigs or {})]
    if unknown:
        logger.warning(f'sensors to skip that are not in the input: {unknown}')
    return kapture_data


def _extract_to_dir(input_path: str,
                    output_path: str,
                    kapture_data: kapture.Kapture,
                    windows: List[Window],
                    time_unit: float,
                    skip_sensor_ids: List[str],
                    transfer: TransferAction,
                    force: bool,
                    points3d: Optional[kapture.Points3d] = None,
                    info: Optional[dict] = None) -> kapture.Kapture:
    """ extracts the windows of kapture_data (loaded from input_path), writes the result and its data files """
    extracted = extract_period(kapture_data, windows, skip_sensor_ids)
    extracted.points3d = points3d

    source_record_dirpath = path.join(input_path, RECORD_DATA_DIRNAME)
    transfer_files = transfer not in (TransferAction.skip, TransferAction.root_link)
    if transfer_files:
        missing = _drop_records_without_file(extracted, source_record_dirpath)
        if missing:
            logger.warning(f'records dropped, their data file is missing in the input: {missing}')

    total = time_range(kapture_data)
    for t0, t1 in windows:
        counts = {name: sum(len(data[t]) for t in data if t0 <= t <= t1)
                  for name, data in _timed_data(extracted).items()}
        counts = ', '.join(f'{n} {name}' for name, n in counts.items() if n)
        from_start = f' (+{(t0 - total[0]) / time_unit:.1f} s)' if total else ''
        logger.info(f'period {t0} .. {t1}{from_start}, {(t1 - t0) / time_unit:.1f} s: {counts or "nothing"}')
    record_counts = {}
    for name, data in _timed_data(kapture_data).items():
        kept = sum(len(of_sensors) for of_sensors in getattr(extracted, name).values())
        record_counts[name] = kept
        logger.info(f'{name}: {kept} / {sum(len(of_sensors) for of_sensors in data.values())}')
    if not any(record_counts.values()):
        logger.warning('nothing in the given periods: the extracted kapture is empty')
    if points3d is not None:
        record_counts['points3d'] = len(points3d)

    os.makedirs(output_path, exist_ok=True)
    delete_existing_kapture_files(output_path, force_erase=force)
    logger.info(f'writing {output_path}')
    kapture_to_dir(output_path, extracted)

    if transfer == TransferAction.root_link:
        if path.isdir(source_record_dirpath):
            import_record_data_from_dir_link_dir(path.realpath(source_record_dirpath), output_path)
    elif transfer_files:
        filenames = sorted({filename for name in FILE_RECORDS_NAMES if getattr(extracted, name) is not None
                            for _, _, filename in kapture.flatten(getattr(extracted, name))})
        transfer_files_from_dir(
            [path.realpath(path.join(source_record_dirpath, filename)) for filename in filenames],
            [get_record_fullpath(output_path, filename) for filename in filenames],
            transfer, force_overwrite=True)
    _transfer_raw(path.join(input_path, RAW_DIRNAME), path.join(output_path, RAW_DIRNAME), transfer)

    with open(path.join(output_path, EXTRACT_FILENAME), 'w') as file:
        json.dump({'src': path.abspath(input_path),
                   'windows': windows,
                   **(info or {}),
                   'skip_sensors': skip_sensor_ids,
                   'transfer': transfer.name,
                   'records': record_counts}, file, indent=1)
    return extracted


def kapture_extract_period(input_path: str,
                           output_path: str,
                           windows: Iterable[Sequence[float]],
                           relative: bool = False,
                           time_unit: float = 1e6,
                           skip_sensor_ids: Iterable[str] = (),
                           transfer: TransferAction = TransferAction.skip,
                           force: bool = False) -> kapture.Kapture:
    """
    Writes a new kapture with the poses and the records of some periods of the input kapture.

    :param input_path: input kapture directory.
    :param output_path: output kapture directory.
    :param windows: periods (start, end) to keep, both included: kapture timestamps, or see relative.
    :param relative: the periods are in seconds since the first timestamp of the input.
    :param time_unit: number of kapture timestamp units per second (relative periods only).
    :param skip_sensor_ids: sensors (or rigs) left out.
    :param transfer: how to bring the data files (images, point clouds, depth and range maps) and sensors/raw.
                     The links point to the actual files: extracting from an extraction made of links gives
                     links to the original files.
    :param force: silently overwrite the kapture files of the output directory.
    :return: the extracted kapture.
    """
    _check_paths(input_path, output_path, transfer)
    skip_sensor_ids = sorted(set(skip_sensor_ids))
    kapture_data = _load(input_path, skip_sensor_ids)
    windows = normalize_windows(windows, relative, time_unit, kapture_data)
    return _extract_to_dir(input_path, output_path, kapture_data, windows, time_unit, skip_sensor_ids, transfer, force)


def kapture_extract_area(input_path: str,
                         output_path: str,
                         box: Optional[Sequence[Optional[float]]] = None,
                         sphere: Optional[Sequence[float]] = None,
                         gap: float = 5.,
                         pad: float = 0.2,
                         time_unit: float = 1e6,
                         crop_points3d: bool = False,
                         skip_sensor_ids: Iterable[str] = (),
                         transfer: TransferAction = TransferAction.skip,
                         force: bool = False) -> kapture.Kapture:
    """
    Writes a new kapture with the poses and the records of the periods the trajectory spends in an area:
    a box aligned with the axes of the world frame, or a sphere. No assumption is made on the axes of the world
    frame (which one is vertical). The records are selected by the position of the device, not by what they see:
    the scans and images taken from inside the area are kept whole.

    :param input_path: input kapture directory, with trajectories.
    :param output_path: output kapture directory.
    :param box: (xmin, xmax, ymin, ymax, zmin, zmax) of the area, in the world frame; None for no bound on
                that side (eg. (-10, 10, -10, 10, None, None) does not filter on z).
    :param sphere: (x, y, z, radius) of the area, in the world frame.
    :param gap: two poses inside the area less than gap seconds apart are in the same period.
    :param pad: seconds added before and after every period.
    :param time_unit: number of kapture timestamp units per second.
    :param crop_points3d: also extract the points3d that are inside the area (observations are not extracted).
    :param skip_sensor_ids: sensors (or rigs) left out; a skipped trajectory device does not select periods.
    :param transfer: how to bring the data files and sensors/raw, see kapture_extract_period.
    :param force: silently overwrite the kapture files of the output directory.
    :return: the extracted kapture.
    """
    _check_paths(input_path, output_path, transfer)
    skip_sensor_ids = sorted(set(skip_sensor_ids))
    if (box is None) == (sphere is None):
        raise ValueError('the area is either a box or a sphere')
    if box is not None:
        box = [None if value is None else float(value) for value in box]
        area, info = box_area(box), {'box': box}
    else:
        if len(sphere) != 4:
            raise ValueError('a sphere is (x, y, z, radius)')
        sphere = [float(value) for value in sphere]
        area, info = sphere_area(sphere[:3], sphere[3]), {'sphere': sphere}
    kapture_data = _load(input_path, skip_sensor_ids, with_points3d=crop_points3d)
    windows = windows_in_area(kapture_data, area, gap, pad, time_unit, skip_sensor_ids)
    if not windows:
        raise ValueError('no pose of the trajectory is inside the area')
    nb_poses = len(trajectory_positions(kapture_data.trajectories, skip_sensor_ids)[0])
    logger.info(f'{len(windows)} periods in the area ({nb_poses} poses in the trajectory)')
    points3d = None
    if crop_points3d and kapture_data.points3d is not None:
        points3d = kapture.Points3d(np.asarray(kapture_data.points3d)[area(np.asarray(kapture_data.points3d))])
        logger.info(f'points3d: {len(points3d)} / {len(kapture_data.points3d)}')
    return _extract_to_dir(input_path, output_path, kapture_data, windows, time_unit, skip_sensor_ids, transfer, force,
                           points3d=points3d, info={**info, 'gap': gap, 'pad': pad})
