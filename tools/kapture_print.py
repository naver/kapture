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

import path_to_kapture
import kapture
import kapture.utils.logging
import kapture.io.csv

logger = logging.getLogger('kapture_print')


@contextlib.contextmanager
def open_or_stdout(filename=None):
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


def print_key_value(key, value, file, show_none):
    if value is not None or show_none:
        print(f'{key:25}: {value}', file=file)


def print_title(title, file):
    print(f'=== {title:^25} ===', file=file)


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
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture data root directory.')
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

    # print
    with open_or_stdout(args.output) as output_stream:
        if args.detail:
            print_title('general', file=output_stream)
            print_key_value('path', args.input, file=output_stream, show_none=args.all)
            print_key_value('version', kapture_data.format_version, file=output_stream, show_none=args.all)

        if not args.detail:
            print_key_value('nb sensors', len(kapture_data.sensors), file=output_stream, show_none=args.all)
        else:
            print_title('sensors', file=output_stream)
            cam_ids = [s_id for s_id, cam in kapture_data.sensors.items() if cam.sensor_type == 'camera']
            print_key_value(' ├─ nb cameras', len(cam_ids), file=output_stream, show_none=args.all)
            lidar_ids = [s_id for s_id, cam in kapture_data.sensors.items() if cam.sensor_type == 'lidar']
            print_key_value(' ├─ nb lidar', len(lidar_ids), file=output_stream, show_none=args.all)
            wifi_ids = [s_id for s_id, cam in kapture_data.sensors.items() if cam.sensor_type == 'wifi']
            print_key_value(' ├─ nb wifi', len(wifi_ids), file=output_stream, show_none=args.all)
            gnss_ids = [s_id for s_id, cam in kapture_data.sensors.items() if cam.sensor_type == 'gnss']
            print_key_value(' ├─ nb gnss', len(gnss_ids), file=output_stream, show_none=args.all)
            print_key_value(' └─ nb sensors total', len(kapture_data.sensors), file=output_stream, show_none=args.all)

        # for rigs, count the number of different rigs ids (not sensors in it).
        nb_rigs = len(kapture_data.rigs) if kapture_data.rigs is not None else None
        if not args.detail:
            print_key_value('nb rigs', nb_rigs, file=output_stream, show_none=args.all)
        else:
            print_title('rigs', file=output_stream)
            if kapture_data.rigs is not None:
                for rig_id in kapture_data.rigs:
                    s_ids = [s_id for s_id in kapture_data.rigs[rig_id]]
                    print_key_value(f' ├─ nb sensors in rig "{rig_id}"', len(s_ids), file=output_stream,
                                    show_none=args.all)
            print_key_value(' └─ nb rigs total', nb_rigs, file=output_stream, show_none=args.all)

        # records (+trajectories)
        for record_name in ['trajectories', 'records_camera', 'records_lidar', 'records_wifi', 'records_gnss']:
            record = getattr(kapture_data, record_name)
            nb_record = None if record is None else len(list(kapture.flatten(record)))
            if not args.detail:
                print_key_value(f'nb {record_name} records', nb_record, file=output_stream, show_none=args.all)
            elif record is not None or args.all:
                print_title(f'{record_name}', file=output_stream)
                if record is not None:
                    timestamp_min, timestamp_max = min(record), max(record)
                    nb_sensors = len(set(s_id for _, s_id, _ in kapture.flatten(record)))
                    print_key_value(' ├─ timestamp range', f'{timestamp_min}:{timestamp_max}', file=output_stream,
                                    show_none=args.all)
                    print_key_value(' ├─ nb sensors', f'{nb_sensors}', file=output_stream, show_none=args.all)
                print_key_value(' └─ nb total', nb_record, file=output_stream, show_none=args.all)

        # image features
        for feature_name in ['keypoints', 'descriptors', 'global_features']:
            feature = getattr(kapture_data, feature_name)
            nb_feature = None if feature is None else len(list(feature))
            if not args.detail:
                print_key_value(f'nb {feature_name}', nb_feature, file=output_stream, show_none=args.all)
            elif feature is not None or args.all:
                print_title(feature_name, file=output_stream)
                if feature is not None:
                    print_key_value(' ├─ kind ', feature.type_name, file=output_stream, show_none=args.all)
                    print_key_value(' ├─ data type', feature.dtype.__name__, file=output_stream, show_none=args.all)
                    print_key_value(' ├─ data size', feature.dsize, file=output_stream, show_none=args.all)
                print_key_value(' └─ nb images', len(kapture_data.keypoints), file=output_stream, show_none=args.all)

        # matches
        nb_matches = None if kapture_data.matches is None else len(list(kapture_data.matches))
        if not args.detail:
            print_key_value('nb matching pairs', nb_matches, file=output_stream, show_none=args.all)
        elif kapture_data.matches is not None or args.all:
            print_title('matches', file=output_stream)
            print_key_value(' └─ nb pairs', nb_matches, file=output_stream, show_none=args.all)

        # points 3D
        nb_points3d = None if kapture_data.points3d is None else len(list(kapture_data.points3d))
        if not args.detail:
            print_key_value('nb points 3-D', nb_points3d, file=output_stream, show_none=args.all)
        elif kapture_data.points3d is not None or args.all:
            print_title('points 3-D', file=output_stream)
            print_key_value(' └─ nb points 3-D', nb_points3d, file=output_stream, show_none=args.all)

        # observations
        nb_observations_3d = len(kapture_data.observations) if kapture_data.observations is not None else None
        nb_observations_2d = len([feat
                                  for feats in kapture_data.observations.values()
                                  for feat in feats]) if kapture_data.observations is not None else None
        if not args.detail:
            print_key_value('nb observed 3-D points', nb_observations_3d, file=output_stream, show_none=args.all)
            print_key_value('nb observation 2-D points', nb_observations_2d, file=output_stream, show_none=args.all)
        elif kapture_data.observations is not None or args.all:
            print_title('Observations', file=output_stream)
            if kapture_data.observations is not None:
                print_key_value(' ├─ nb observed 3-D', nb_observations_3d, file=output_stream, show_none=args.all)
            print_key_value(' └─ nb observations 2-D', nb_observations_2d, file=output_stream, show_none=args.all)


if __name__ == '__main__':
    print_command_line()
