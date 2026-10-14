#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data to Long-term Visual Localization challenge format

This line should store the
result as `name.jpg qw qx qy qz tx ty tz`.
Here,  `name` corresponds to the filename of the image, without any directory names.
`qw qx qy qz` represents the **rotation** from world to camera coordinates as a
**unit quaternion**. `tx ty tz` is the camera **translation** (**not the camera position**).

"""
import logging
import os
import os.path as path
import pathlib
from tqdm import tqdm

import kapture
import kapture.io.features
import kapture.io.csv
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('LTVL2020')


def export_ltvl(input: str,
                output: str,
                full_file_name: bool = False,
                prepend_cam: bool = False,
                truncate_extensions: bool = False,
                inverse_pose: bool = False,
                force: bool = False,
                verbose: Verbosity = DEFAULT_VERBOSITY,
                quiet: bool = False) -> None:
    """
    Export kapture data to a Long-term Visual Localization challenge format file
    (https://www.visuallocalization.net/submission/).
    With --prepend_cam, it is identical to the RobotCar_Seasons format for https://www.visuallocalization.net/
    with --full_file_name, it is identical to the Gangnam Station and Hyundai Department Store format
    With --full_file_name, --truncate_extensions and --inverse_pose it is identical to the rio10 submission format
    (http://vmnavab26.in.tum.de/RIO10/)

    :param input: kapture data top directory
    :param output: LTVL file path to write
    :param full_file_name: 1st operation: if False, basename is kept for image name (seq-0/image1.png -> image1.png),
                           if True, full file name is preserved.
    :param prepend_cam: 2nd operation: if True, prepend the camera name to the image file names
                        (required for some dataset).
    :param truncate_extensions: 3rd operation: if True, name is image_name before the first dot.
    :param inverse_pose: write pose as camera to world instead of world to camera.
    :param force: accepted for compatibility, unused (output is always overwritten).
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    # only load (1) image records + (2) trajectories (that's all that matters).
    # 1: load records
    records_camera_filepath = kapture.io.csv.get_csv_fullpath(kapture.RecordsCamera, input)
    logger.debug(f'loading {records_camera_filepath}')
    records_cameras = kapture.io.csv.records_camera_from_file(records_camera_filepath)
    # 2: load trajectories
    trajectories_filepath = kapture.io.csv.get_csv_fullpath(kapture.Trajectories, input)
    logger.debug(f'loading {trajectories_filepath}')
    trajectories = kapture.io.csv.trajectories_from_file(trajectories_filepath)

    # 2.2: load rigs if it exists
    rigs_filepath = kapture.io.csv.get_csv_fullpath(kapture.Rigs, input)
    if path.isfile(rigs_filepath):
        logger.debug(f'loading {rigs_filepath}')
        rigs = kapture.io.csv.rigs_from_file(rigs_filepath)
        trajectories = kapture.rigs_remove(trajectories, rigs)

    # 3: find (timestamp, camera_id) that are both in records and trajectories.
    valid_keys = set(records_cameras.key_pairs()).intersection(set(trajectories.key_pairs()))
    # collect data for those timestamps.
    if full_file_name:
        image_poses = ((k[1], records_cameras[k], trajectories[k]) for k in valid_keys)
    else:
        image_poses = ((k[1], path.basename(records_cameras[k]), trajectories[k]) for k in valid_keys)

    # prepend the camera name or drop it.
    if prepend_cam:
        image_poses = ((path.join(camera_id, image_filename), pose) for camera_id, image_filename, pose in image_poses)
    else:
        image_poses = ((image_filename, pose) for _, image_filename, pose in image_poses)

    if truncate_extensions:
        image_poses = ((image_filename[:image_filename.index('.')], pose) for image_filename, pose in image_poses)

    # write the files
    image_poses = {image_filename: pose
                   for image_filename, pose in image_poses}
    p = pathlib.Path(output)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output, 'wt') as f:
        for image_filename, pose in tqdm(image_poses.items(), disable=logger.getEffectiveLevel() > logging.INFO):
            if inverse_pose:
                pose = pose.inverse()
            line = [image_filename] + pose.r_raw + pose.t_raw
            line = ' '.join(str(v) for v in line) + '\n'
            f.write(line)


def main():
    run_cli(export_ltvl, aliases={'--prepend_cam': ['-p']})


if __name__ == '__main__':
    main()
