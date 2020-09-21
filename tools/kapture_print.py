#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to print statistics about kapture data.
"""

import argparse
import logging
import sys
import contextlib
import os.path as path

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.csv

logger = logging.getLogger('kapture_print')


@contextlib.contextmanager
def open_or_stdout(filename=None):
    """
    Get the output to print into
    """
    # from https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def print_key_value(key, value, file, show_none) -> None:
    """
    Prints a key and its value
    """
    if value is not None or show_none:
        print(f'{key:25}: {value}', file=file)


def print_title(title, file) -> None:
    """
    Prints the title
    """
    print(f'=== {title:^25} ===', file=file)


def print_sensors(kapture_data, output_stream, show_detail, show_all) -> None:
    """
    Prints the sensors (and rigs) to the output stream
    """
    if not show_detail:
        print_key_value('nb sensors', len(kapture_data.sensors), file=output_stream, show_none=show_all)
    else:
        print_title('sensors', file=output_stream)
        cam_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'camera']
        print_key_value(' ├─ nb cameras', len(cam_ids), file=output_stream, show_none=show_all)
        lidar_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'lidar']
        print_key_value(' ├─ nb lidar', len(lidar_ids), file=output_stream, show_none=show_all)
        wifi_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'wifi']
        print_key_value(' ├─ nb wifi', len(wifi_ids), file=output_stream, show_none=show_all)
        bluetooth_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'bluetooth']
        print_key_value(' ├─ nb bluetooth', len(bluetooth_ids), file=output_stream, show_none=show_all)
        gnss_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'gnss']
        print_key_value(' ├─ nb gnss', len(gnss_ids), file=output_stream, show_none=show_all)
        accelerometer_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'accelerometer']
        print_key_value(' ├─ nb accelerometer', len(accelerometer_ids), file=output_stream, show_none=show_all)
        gyroscope_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'gyroscope']
        print_key_value(' ├─ nb gyroscope', len(gyroscope_ids), file=output_stream, show_none=show_all)
        magnetic_ids = [s_id for s_id, sensor in kapture_data.sensors.items() if sensor.sensor_type == 'magnetic']
        print_key_value(' ├─ nb magnetic', len(magnetic_ids), file=output_stream, show_none=show_all)
        print_key_value(' └─ nb sensors total', len(kapture_data.sensors), file=output_stream, show_none=show_all)
    # for rigs, count the number of different rigs ids (not sensors in it).
    nb_rigs = len(kapture_data.rigs) if kapture_data.rigs is not None else None
    if not show_detail:
        print_key_value('nb rigs', nb_rigs, file=output_stream, show_none=show_all)
    else:
        print_title('rigs', file=output_stream)
        if kapture_data.rigs is not None:
            for rig_id in kapture_data.rigs:
                s_ids = [s_id for s_id in kapture_data.rigs[rig_id]]
                print_key_value(f' ├─ nb sensors in rig "{rig_id}"', len(s_ids), file=output_stream,
                                show_none=show_all)
        print_key_value(' └─ nb rigs total', nb_rigs, file=output_stream, show_none=show_all)


def print_records(kapture_data, output_stream, show_detail, show_all) -> None:
    """
    Prints the records and trajectories to the output stream
    """
    # records (+trajectories)
    for record_name in ['trajectories', 'records_camera', 'records_lidar', 'records_wifi', 'records_bluetooth',
                        'records_gnss', 'records_accelerometer', 'records_gyroscope', 'records_magnetic']:
        record = getattr(kapture_data, record_name)
        nb_record = None if record is None else len(list(kapture.flatten(record)))
        if not show_detail:
            print_key_value(f'nb {record_name}', nb_record, file=output_stream, show_none=show_all)
        elif record is not None or show_all:
            print_title(f'{record_name}', file=output_stream)
            if record is not None and len(record) > 0:
                timestamp_min, timestamp_max = min(record), max(record)
                nb_sensors = len(set(s_id for _, s_id, *x in kapture.flatten(record)))
                print_key_value(' ├─ timestamp range', f'{timestamp_min}:{timestamp_max}', file=output_stream,
                                show_none=show_all)
                print_key_value(' ├─ nb sensors', f'{nb_sensors}', file=output_stream, show_none=show_all)
            print_key_value(' └─ nb total', nb_record, file=output_stream, show_none=show_all)


def print_features(kapture_data, output_stream, show_detail, show_all) -> None:
    """
    Prints the features to the output stream
    """
    # image features
    for feature_name in ['keypoints', 'descriptors', 'global_features']:
        feature = getattr(kapture_data, feature_name)
        nb_feature = None if feature is None else len(list(feature))
        if not show_detail:
            print_key_value(f'nb {feature_name}', nb_feature, file=output_stream, show_none=show_all)
        elif feature is not None or show_all:
            print_title(feature_name, file=output_stream)
            if feature is not None:
                print_key_value(' ├─ kind ', feature.type_name, file=output_stream, show_none=show_all)
                print_key_value(' ├─ data type', feature.dtype.__name__, file=output_stream, show_none=show_all)
                print_key_value(' ├─ data size', feature.dsize, file=output_stream, show_none=show_all)
            print_key_value(' └─ nb images', len(kapture_data.keypoints), file=output_stream, show_none=show_all)


def print_matches(kapture_data, output_stream, show_detail, show_all) -> None:
    """
    Prints the matches to the output stream
    """
    # matches
    nb_matches = None if kapture_data.matches is None else len(list(kapture_data.matches))
    if not show_detail:
        print_key_value('nb matching pairs', nb_matches, file=output_stream, show_none=show_all)
    elif kapture_data.matches is not None or show_all:
        print_title('matches', file=output_stream)
        print_key_value(' └─ nb pairs', nb_matches, file=output_stream, show_none=show_all)


def print_points(kapture_data, output_stream, show_detail, show_all) -> None:
    """
    Prints the 3D points (and observations) to the output stream
    """
    # points 3D
    nb_points3d = None if kapture_data.points3d is None else len(list(kapture_data.points3d))
    if not show_detail:
        print_key_value('nb points 3-D', nb_points3d, file=output_stream, show_none=show_all)
    elif kapture_data.points3d is not None or show_all:
        print_title('points 3-D', file=output_stream)
        print_key_value(' └─ nb points 3-D', nb_points3d, file=output_stream, show_none=show_all)
    # observations
    nb_observations_3d = len(kapture_data.observations) if kapture_data.observations is not None else None
    nb_observations_2d = len([feat
                              for feats in kapture_data.observations.values()
                              for feat in feats]) if kapture_data.observations is not None else None
    if not show_detail:
        print_key_value('nb observed 3-D points', nb_observations_3d, file=output_stream, show_none=show_all)
        print_key_value('nb observation 2-D points', nb_observations_2d, file=output_stream, show_none=show_all)
    elif kapture_data.observations is not None or show_all:
        print_title('Observations', file=output_stream)
        if kapture_data.observations is not None:
            print_key_value(' ├─ nb observed 3-D', nb_observations_3d, file=output_stream, show_none=show_all)
        print_key_value(' └─ nb observations 2-D', nb_observations_2d, file=output_stream, show_none=show_all)


def print_command_line() -> None:
    """
    Do the print using the parameters given on the command line.
    """

    parser = argparse.ArgumentParser(description='Print on stdout (or given file) statistics about kapture dataset.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', '-k', '--kapture', required=True,
                        help='path to kapture data root directory.')
    parser.add_argument('-o', '--output', required=False, default='-',
                        help='output file [stdout]')
    parser.add_argument('-a', '--all', action='store_true', default=False,
                        help='display all, even None')
    parser.add_argument('-d', '--detail', action='store_true', default=False,
                        help='display detailed')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    args.input = path.abspath(args.input)
    # load
    kapture_data = kapture.io.csv.kapture_from_dir(args.input)
    do_print(kapture_data, args.input, args.output, args.detail, args.all)


def do_print(kapture_data: kapture, kapture_name: str, output: str, show_detail: bool, show_all: bool) -> None:
    """
    Do the print using the given user parameters
    """

    # print
    with open_or_stdout(output) as output_stream:
        if show_detail:
            print_title('general', file=output_stream)
            print_key_value('path', kapture_name, file=output_stream, show_none=show_all)
            print_key_value('version', kapture_data.format_version, file=output_stream, show_none=show_all)

        print(kapture_data)
        print_sensors(kapture_data, output_stream, show_detail, show_all)
        print_records(kapture_data, output_stream, show_detail, show_all)
        print_features(kapture_data, output_stream, show_detail, show_all)
        print_matches(kapture_data, output_stream, show_detail, show_all)
        print_points(kapture_data, output_stream, show_detail, show_all)


if __name__ == '__main__':
    print_command_line()
