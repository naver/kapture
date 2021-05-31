#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to print statistics about kapture data.
"""

import argparse
import logging
import os
import os.path as path
import re
from datetime import datetime

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.computation as computation
import kapture.utils.logging
import kapture.io.csv

logger = logging.getLogger('kapture_data_date')

KAPTURE_FILE_PARSING_RE = re.compile('.*(records|trajectories).*\\.txt')


def format_timestamp_range(first_timestamp: int, last_timestamp: int) -> str:
    """
    Format a time range with unit.

    :param first_timestamp: first timestamp
    :param last_timestamp: last timestamp
    :return string: formatted time range
    """
    timestamp_len = 1
    try:
        timestamp_len = computation.num_digits(first_timestamp)
        delta_len = 10 - timestamp_len  # datetime expects epoch timestamp in seconds
        factor = 10**delta_len
        dts = [datetime.utcfromtimestamp(first_timestamp * factor),
               datetime.utcfromtimestamp(last_timestamp * factor)]
        timestamp_format_d = {
            'date': '%Y/%m/%d',
            'time': '%H:%M:%S.%f'
        }
        timestamp_parts = [
            {
                pname: dt.strftime(pformat)
                for pname, pformat in timestamp_format_d.items()
            }
            for dt in dts
        ]
        timestamp_str = timestamp_parts[0]['date'] + ' ' + timestamp_parts[0]['time']
        timestamp_str += ' -> '
        if timestamp_parts[0]['date'] != timestamp_parts[1]['date']:
            timestamp_str = timestamp_parts[0]['date'] + ' '
        timestamp_str += timestamp_parts[1]['time'] + ' GMT'
        return timestamp_str
    except ValueError as _:  # noqa: F841
        return f'{first_timestamp} -> {last_timestamp}' \
               + f' ** FAIL to parse as posix timestamp of {timestamp_len} digits'


def print_info(kapture_path: str, kapture_filename: str) -> None:
    """
    Prints some info on given file that should contains records with timestamps

    :param kapture_path: kapture top directory
    :param kapture_filename: full path of a valid kapture file
    """
    # Check version to test if this is a real kapture file
    kapture_file_path = path.join(kapture_path, kapture_filename)
    version = kapture.io.csv.get_version_from_csv_file(kapture_file_path)
    if version is None:
        logger.debug(f'{kapture_filename} not a kapture file')
    else:
        # Read the file
        last_data_line = ''
        nb_lines = 0
        with open(kapture_file_path) as f:
            # Count lines
            for _ in f:
                nb_lines += 1
            # Reset to read some important lines
            f.seek(0, os.SEEK_SET)
            # Skip header
            f.readline()
            f.readline()
            first_data_line = f.readline()
            if first_data_line:
                last_data_line = kapture.io.csv.get_last_line(f)
        first_timestamp = 0
        last_timestamp = 0
        # data line are comma separated lines with the timestamp as first value
        try:
            if first_data_line:
                first_timestamp = int(first_data_line.split(',')[0])
            if last_data_line:
                last_timestamp = int(last_data_line.split(',')[0])
        except ValueError:
            pass
        if first_timestamp > 0 and last_timestamp > 0:
            timestamp_range_str = format_timestamp_range(first_timestamp, last_timestamp)
            timestamp_len1 = computation.num_digits(first_timestamp)
            timestamp_len2 = computation.num_digits(last_timestamp)
            timestamp_len_str = f'{timestamp_len1}' if timestamp_len1 == timestamp_len2\
                else f'{timestamp_len1}-{timestamp_len2}'
            print(f'{kapture_filename:42s} timestamp {timestamp_len_str} digits from {timestamp_range_str}'
                  f' : {(nb_lines-2):12,d} records'.replace(',', ' '))


def do_print(kapture_path: str) -> None:
    """
    Print out kapture data:

    :param kapture_path: full path to kapture directory.
    """
    dirs_to_examine = set()
    for k_path in kapture.io.csv.CSV_FILENAMES.values():
        dirs_to_examine.add(path.dirname(k_path))
    # Look for files which might contain kapture data
    for k_dir in sorted(dirs_to_examine):
        kapture_dir = path.join(kapture_path, k_dir)
        if path.exists(kapture_dir):
            if path.isdir(kapture_dir):
                # print
                logger.debug(f'browsing directory {k_dir}')
                # Search for records or trajectory like files
                kapture_files = []
                for entry in os.listdir(kapture_dir):
                    if path.isfile(path.join(kapture_dir, entry)):
                        if KAPTURE_FILE_PARSING_RE.match(entry) is not None:
                            kapture_files.append(entry)
                for kapture_file in sorted(kapture_files):
                    print_info(kapture_path, path.join(k_dir, kapture_file))
            else:
                logger.fatal(f'{kapture_dir} is not a directory')
        else:
            logger.debug(f'{kapture_dir} does not exist')


def print_command_line() -> None:
    """
    Do the print using the parameters given on the command line.
    """

    parser = argparse.ArgumentParser(description='Print statistics about kapture records files with timestamps.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-k', '--kapture', required=True,
                        help='path to kapture data root directory.')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    do_print(path.abspath(args.kapture))


if __name__ == '__main__':
    print_command_line()
