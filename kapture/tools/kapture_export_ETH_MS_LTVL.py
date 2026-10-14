#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data to ETH-Microsoft Long-term Visual Localization challenge format
See https://github.com/cvg/visloc-iccv2021

This line should store the
result as `timestamp/camera_or_rig_id qw qx qy qz tx ty tz`.
`qw qx qy qz` represents the **rotation** from world to camera coordinates as a
**unit quaternion**. `tx ty tz` is the camera **translation** (**not the camera position**).

"""
import logging
import os
import pathlib
from tqdm import tqdm

import kapture
from kapture.core.Trajectories import rigs_recover
import kapture.io.features
import kapture.io.csv
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('ETH_MS_LTVL')


def export_ETH_MS_LTVL(input: str,
                       output: str,
                       force: bool = False,
                       verbose: Verbosity = DEFAULT_VERBOSITY,
                       quiet: bool = False) -> None:
    """
    Export kapture data to ETH-Microsoft Long-term Visual Localization challenge format file
    (https://www.visuallocalization.net/submission/).

    :param input: kapture data top directory
    :param output: LTVL file path to write
    :param force: accepted for compatibility, unused (output is always overwritten).
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    skip_heavy = [kapture.RecordsLidar, kapture.RecordsWifi,
                  kapture.Keypoints, kapture.Descriptors,
                  kapture.GlobalFeatures, kapture.Matches,
                  kapture.Points3d, kapture.Observations]
    kapture_data = kapture.io.csv.kapture_from_dir(input, skip_list=skip_heavy)

    # no nested rigs in ETH-MS
    assert kapture_data.rigs is not None
    assert kapture_data.trajectories is not None
    trajectories = rigs_recover(kapture_data.trajectories, kapture_data.rigs)

    # write the file

    p = pathlib.Path(output)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output, 'wt') as f:
        for timestamp, sensor_id, pose in tqdm(kapture.flatten(trajectories),
                                               disable=logger.getEffectiveLevel() > logging.INFO):
            line = [str(timestamp) + '/' + sensor_id] + pose.r_raw + pose.t_raw
            line = ' '.join(str(v) for v in line) + '\n'
            f.write(line)


def main():
    run_cli(export_ETH_MS_LTVL)


if __name__ == '__main__':
    main()
