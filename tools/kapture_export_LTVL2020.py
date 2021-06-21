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
import argparse
from tqdm import tqdm

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.features
import kapture.io.csv

logger = logging.getLogger('LTVL2020')


def export_ltvl(kapture_dirpath: str,
                ltvl_filepath: str,
                keep_full_file_name: bool = False,
                prepend_camera_name: bool = False,
                truncate_extensions: bool = False,
                inverse_pose: bool = False) -> None:
    """
    Export kapture data to a Long-term Visual Localization challenge format file.
    With -p, it is identical to the RobotCar_Seasons format for https://www.visuallocalization.net/
    with --keep_full_file_name, it is identical to the Gangnam Station and Hyundai Department Store format
    With --keep_full_file_name, --truncate_extensions and --inverse-pose it is identical to the rio10 submission format

    :param kapture_dirpath: kapture data top directory
    :param ltvl_filepath: LTVL file path to write
    :param keep_full_file_name: 1st operation, if False, basename is kept for image name
    :param prepend_camera_name: 2nd operation, if True, it will prepend the camera name to the image file names
    :param truncate_extensions: 3rd operation, if True, name is image_name before the first dot
    """
    # only load (1) image records + (2) trajectories (that all that matters).
    # 1: load records
    records_camera_filepath = kapture.io.csv.get_csv_fullpath(kapture.RecordsCamera, kapture_dirpath)
    logger.debug(f'loading {records_camera_filepath}')
    records_cameras = kapture.io.csv.records_camera_from_file(records_camera_filepath)
    # 2: load trajectories
    trajectories_filepath = kapture.io.csv.get_csv_fullpath(kapture.Trajectories, kapture_dirpath)
    logger.debug(f'loading {trajectories_filepath}')
    trajectories = kapture.io.csv.trajectories_from_file(trajectories_filepath)

    # 2.2: load rigs if it exists
    rigs_filepath = kapture.io.csv.get_csv_fullpath(kapture.Rigs, kapture_dirpath)
    if path.isfile(rigs_filepath):
        logger.debug(f'loading {rigs_filepath}')
        rigs = kapture.io.csv.rigs_from_file(rigs_filepath)
        trajectories = kapture.rigs_remove(trajectories, rigs)

    # 3: find (timestamp, camera_id) that are both in records and trajectories.
    valid_keys = set(records_cameras.key_pairs()).intersection(set(trajectories.key_pairs()))
    # collect data for those timestamps.
    if keep_full_file_name:
        image_poses = ((k[1], records_cameras[k], trajectories[k]) for k in valid_keys)
    else:
        image_poses = ((k[1], path.basename(records_cameras[k]), trajectories[k]) for k in valid_keys)

    # prepend the camera name or drop it.
    if prepend_camera_name:
        image_poses = ((path.join(camera_id, image_filename), pose) for camera_id, image_filename, pose in image_poses)
    else:
        image_poses = ((image_filename, pose) for _, image_filename, pose in image_poses)

    if truncate_extensions:
        image_poses = ((image_filename[:image_filename.index('.')], pose) for image_filename, pose in image_poses)

    # write the files
    image_poses = {image_filename: pose
                   for image_filename, pose in image_poses}
    p = pathlib.Path(ltvl_filepath)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(ltvl_filepath, 'wt') as f:
        for image_filename, pose in tqdm(image_poses.items(), disable=logger.getEffectiveLevel() > logging.INFO):
            if inverse_pose:
                pose = pose.inverse()
            line = [image_filename] + pose.r_raw + pose.t_raw
            line = ' '.join(str(v) for v in line) + '\n'
            f.write(line)


def export_ltvl2020_command_line() -> None:
    """
    Do the LTVL 2020 export using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(
        description=('convert file to Long-term Visual Localization challenge format '
                     '(https://www.visuallocalization.net/submission/). '
                     'With -p, it is identical to the RobotCar_Seasons format for https://www.visuallocalization.net/ '
                     'with --keep_full_file_name, it is identical to the Gangnam Station '
                     'and Hyundai Department Store format '
                     '  --keep_full_file_name, --truncate_extensions and --inverse-pose '
                     'it is identical to the rio10 submission format (http://vmnavab26.in.tum.de/RIO10/)'))
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

    parser.add_argument('--full_file_name', action='store_true', default=False,
                        help=('1st operation applied: '
                              'when False (default), basename is kept for image name: (seq-0/image1.png -> image1.png).'
                              'when True, full file name is preserved'))
    parser.add_argument('-p', '--prepend_cam', action='store_true', default=False,
                        help=('2nd operation applied: '
                              'when False (default), nothing is changed.'
                              'wen True, prepend camera names to filename (required for some dataset)'))
    parser.add_argument('--truncate-extensions', action='store_true', default=False,
                        help=('2nd operation applied: '
                              'when False (default), nothing is changed.'
                              'when True, name is image_name before the first dot'))
    parser.add_argument('--inverse-pose', action='store_true', default=False,
                        help=('write pose as camera to world instead of world to camera'))
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    export_ltvl(args.input, args.output, args.full_file_name, args.prepend_cam, args.truncate_extensions,
                args.inverse_pose)


if __name__ == '__main__':
    export_ltvl2020_command_line()
