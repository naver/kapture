# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Mainly about rig calibration
"""

from tqdm import tqdm
from typing import Optional, List
import logging
from kapture import Trajectories, Rigs
from kapture.utils.logging import getLogger

logger = getLogger()


def rigs_calibrate_average(
        trajectories: Trajectories,
        rigs: Rigs,
        master_sensors: Optional[List[str]] = None
) -> Rigs:
    """
    Recomputes rig calibrations, ie. the poses of the senors inside a rig.
    It uses the poses of the senors in the given trajectories, to average a average rig.

    :param trajectories:
    :param rigs:
    :param master_sensors:
    :return:
    """
    # senor_id -> rig_id
    sensors_to_rigs = {sensor_id: rig_id
                       for rig_id, sensor_id in rigs.key_pairs()}

    show_progress_bar = logger.getEffectiveLevel() <= logging.INFO
    for timestamp, poses in tqdm(trajectories.items(), disable=not show_progress_bar):
        # accumulate relative poses inside rig
        for sensor_id, pose_sensor_from_world in poses.items():
            pass

    rigs_calibrated = Rigs()
    return rigs_calibrated
