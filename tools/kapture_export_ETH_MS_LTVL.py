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
import argparse
from tqdm import tqdm

import path_to_kapture  # noqa: F401
import kapture
from kapture.core.Trajectories import rigs_recover
import kapture.utils.logging
import kapture.io.features
import kapture.io.csv

logger = logging.getLogger('ETH_MS_LTVL')


def export_ETH_MS_LTVL(kapture_dirpath: str,
                       ltvl_filepath: str) -> None:
    """
    Export kapture data to ETH-Microsoft Long-term Visual Localization challenge format file.

    :param kapture_dirpath: kapture data top directory
    :param ltvl_filepath: LTVL file path to write

    """
    skip_heavy = [kapture.RecordsLidar, kapture.RecordsWifi,
                  kapture.Keypoints, kapture.Descriptors,
                  kapture.GlobalFeatures, kapture.Matches,
                  kapture.Points3d, kapture.Observations]
    kapture_data = kapture.io.csv.kapture_from_dir(kapture_dirpath, skip_list=skip_heavy)

    # no nested rigs in ETH-MS
    assert kapture_data.rigs is not None
    assert kapture_data.trajectories is not None
    trajectories = rigs_recover(kapture_data.trajectories, kapture_data.rigs)

    # write the file

    p = pathlib.Path(ltvl_filepath)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(ltvl_filepath, 'wt') as f:
        for timestamp, sensor_id, pose in tqdm(kapture.flatten(trajectories),
                                               disable=logger.getEffectiveLevel() > logging.INFO):
            line = [str(timestamp) + '/' + sensor_id] + pose.r_raw + pose.t_raw
            line = ' '.join(str(v) for v in line) + '\n'
            f.write(line)


def export_ETH_MS_LTVL_command_line() -> None:
    """
    Do the ETH-Microsoft LTVL export using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description=('convert file to ETH-Microsoft Long-term Visual Localization challenge format '
                     '(https://www.visuallocalization.net/submission/).'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # export ###########################################################################################################
    parser.add_argument('-i', '--input', required=True, help='input path to kapture directory')
    parser.add_argument('-o', '--output', required=True, help='output file.')

    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    export_ETH_MS_LTVL(args.input, args.output)


if __name__ == '__main__':
    export_ETH_MS_LTVL_command_line()
