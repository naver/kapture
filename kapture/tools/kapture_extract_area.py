#!/usr/bin/env python3
# Copyright 2026-present NAVER Corp. Under BSD 3-clause license

"""
This script extracts the part of a kapture recording made in an area to a new kapture.
"""

from typing import List, Optional, Union

from kapture.algo.extract import kapture_extract_area, logger
from kapture.io.records import TransferAction
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY


NO_BOUND = 'NONE'
""" keyword to give instead of a value of --box for no bound on that side """


def _box_bound(value: Union[float, str]) -> Optional[float]:
    """ value of --box: a number, or the keyword NONE (any case) """
    if isinstance(value, str):
        if value.strip().upper() == NO_BOUND:
            return None
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'--box expects numbers or {NO_BOUND}, got "{value}"')
    return float(value)


def extract_area(input: str,
                 output: str,
                 box: Optional[List[Union[float, str]]] = None,
                 sphere: Optional[List[float]] = None,
                 gap: float = 5.,
                 pad: float = 0.2,
                 time_unit: float = 1e6,
                 crop_points3d: bool = False,
                 skip_sensors: List[str] = [],
                 transfer: TransferAction = TransferAction.skip,
                 force: bool = False,
                 verbose: Verbosity = DEFAULT_VERBOSITY,
                 quiet: bool = False) -> None:
    """
    Extract the part of a kapture recording made in an area to a new kapture.

    The area is a box aligned with the axes of the world frame, or a sphere. Keeps the sensors, the rigs,
    and the poses and records (images, lidar, IMU, wifi, ...) of the periods the trajectory spends in the area: the
    records are selected by the position of the device, not by what they see. Warning: the data files are not
    transferred by default (transfer=skip). The area, the periods and the number of records kept are written in
    <output>/subset.json.

    :param input: path to the top directory of the kapture to extract from (with trajectories)
    :param output: path to the top directory of the kapture to create
    :param box: area as a box of the world frame: --box XMIN XMAX YMIN YMAX ZMIN ZMAX. NONE instead of a value
                means no bound on that side: --box -10 10 -10 10 NONE NONE does not filter on z.
    :param sphere: area as a sphere of the world frame: --sphere X Y Z RADIUS
    :param gap: two poses inside the area less than gap seconds apart are in the same period (a short way out of
                the area does not cut the period)
    :param pad: seconds kept before and after every period
    :param time_unit: number of kapture timestamp units per second (1e6: microseconds), for gap and pad
    :param crop_points3d: also extract the points3d that are inside the area (observations are not extracted)
    :param skip_sensors: ids of the sensors to leave out
    :param transfer: how to bring the data files (images, point clouds, depth / range maps, sensors/raw). The links
                     point to the actual files, even when the input is itself made of links.
    :param force: silently overwrite kapture files if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    if (box is None) == (sphere is None):
        raise ValueError('give the area with one of --box, --sphere')
    if box is not None:
        if len(box) != 6:
            raise ValueError(f'--box expects "xmin xmax ymin ymax zmin zmax", got {len(box)} values')
        box = [_box_bound(value) for value in box]
    if sphere is not None and len(sphere) != 4:
        raise ValueError(f'--sphere expects "x y z radius", got {len(sphere)} values')
    kapture_extract_area(input, output,
                         box=box,
                         sphere=sphere,
                         gap=gap,
                         pad=pad,
                         time_unit=time_unit,
                         crop_points3d=crop_points3d,
                         skip_sensor_ids=skip_sensors,
                         transfer=transfer,
                         force=force)


def main():
    run_cli(extract_area, aliases={'--box': ['-b'], '--skip_sensors': ['-s']})


if __name__ == '__main__':
    main()
