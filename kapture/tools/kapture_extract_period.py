#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
This script extracts some periods of time of a kapture recording to a new kapture.
"""

from typing import List

from kapture.algo.extract import kapture_extract_period, logger
from kapture.io.records import TransferAction
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


def extract_period(input: str,
                   output: str,
                   windows: List[float],
                   relative: bool = False,
                   time_unit: float = 1e6,
                   skip_sensors: List[str] = [],
                   transfer: TransferAction = TransferAction.skip,
                   force: bool = False,
                   verbose: Verbosity = DEFAULT_VERBOSITY,
                   quiet: bool = False) -> None:
    """
    Extract some periods of time of a kapture recording to a new kapture.

    Keeps the sensors, the rigs, and the poses and records (images, lidar, IMU, wifi, ...) whose timestamp is in one
    of the periods. The reconstruction data (features, matches, points3d, observations) are not extracted. Warning:
    the data files are not transferred by default (transfer=skip). The periods and the number of records kept are
    written in <output>/subset.json.

    :param input: path to the top directory of the kapture to extract from
    :param output: path to the top directory of the kapture to create
    :param windows: periods to keep, as pairs "start end" (both included): --windows T0 T1 [T0 T1 ...].
                    Kapture timestamps, or seconds since the first timestamp of the recording with --relative.
    :param relative: the periods are given in seconds since the first timestamp of the recording
    :param time_unit: number of kapture timestamp units per second (1e6: microseconds), for --relative
    :param skip_sensors: ids of the sensors to leave out
    :param transfer: how to bring the data files (images, point clouds, depth / range maps, sensors/raw). The links
                     point to the actual files, even when the input is itself made of links.
    :param force: silently overwrite kapture files if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    if len(windows) % 2 != 0:
        raise ValueError(f'--windows expects pairs "start end", got {len(windows)} values')
    kapture_extract_period(input, output,
                           windows=list(zip(windows[0::2], windows[1::2])),
                           relative=relative,
                           time_unit=time_unit,
                           skip_sensor_ids=skip_sensors,
                           transfer=transfer,
                           force=force)


def main():
    run_cli(extract_period, aliases={'--windows': ['-w', '--window'], '--skip_sensors': ['-s']})


if __name__ == '__main__':
    main()
