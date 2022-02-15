#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export 4seasons relative pose between the source and the query frame.
Note that the relative pose is from cam0 of the reference sequence to cam0 of the query sequence, respectively.
see more details at:
https://github.com/pmwenzel/mlad-iccv2021#task
"""

import argparse
import logging
import os.path as path
import numpy as np
import quaternion
from tqdm import tqdm

# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.csv as csv
import kapture.io.features
from kapture.utils.logging import getLogger

logger = getLogger()

q = quaternion.from_rotation_matrix(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]))
CAM_AXES_KAPTURE_FROM_4SEASONS = kapture.PoseTransform(r=q)
CAM_AXES_4SEASONS_FROM_KAPTURE = CAM_AXES_KAPTURE_FROM_4SEASONS.inverse()


def load_timestamp_pairs(pairs_file_path: str):
    t = np.loadtxt(pairs_file_path, dtype=int)
    pairs = t.tolist()
    return pairs


def heal_timestamps_inplace(records_camera: kapture.RecordsCamera,
                            trajectories: kapture.Trajectories):
    # retrieve timestamp/cam_id from filename
    timestamp_old_to_healed = {}
    for timestamp_kapture, cam_id_kapture, image_path in kapture.flatten(records_camera):
        # timestamp
        timestamp_healed = path.basename(image_path)
        timestamp_healed = path.splitext(timestamp_healed)[0]
        timestamp_healed = int(timestamp_healed)
        timestamp_old_to_healed[timestamp_kapture] = timestamp_healed

    for timestamp_kapture, timestamp_healed in timestamp_old_to_healed.items():
        records_camera[timestamp_healed] = records_camera.pop(timestamp_kapture)
        if timestamp_kapture in trajectories:
            trajectories[timestamp_healed] = trajectories.pop(timestamp_kapture)


class NotLocalizedHalt(IndexError):
    pass


def export_4seasons_pairfile(kapture_dir_path: str,
                             pairs_file_path: str,
                             poses_file_path: str,
                             heal_timestamps: bool = True):
    """
    :param kapture_dir_path: input path to kapture directory that contains both localized and mapping.
    :param pairs_file_path:  input path to timestamp pair file.
    :param poses_file_path: output path to relative pose file.
    :param heal_timestamps: overwrite timestamps from filenames.
    :return:
    """
    logger.info('loading kapture data ...')
    with csv.get_all_tar_handlers(kapture_dir_path) as tar_handlers:
        kapture_data = csv.kapture_from_dir(kapture_dir_path, tar_handlers=tar_handlers)

    if heal_timestamps:
        logger.debug('recovering timestamps from filenames.')
        heal_timestamps_inplace(
            kapture_data.records_camera,
            kapture_data.trajectories)

    pairs = load_timestamp_pairs(pairs_file_path)
    logger.info('removing rigs')
    kapture.rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)

    poses = []
    logger.info('Processing poses')
    hide_progress = logger.getEffectiveLevel() > logging.INFO
    for mapping_ts, query_ts in tqdm(pairs, disable=hide_progress):
        try:
            if mapping_ts not in kapture_data.trajectories:
                # mapping, means its probab
                raise IndexError(f'No pose available in trajectory for mapping at time {mapping_ts}. '
                                 f'Have you merged mapping and query in the same kapture dataset ?')

            if query_ts not in kapture_data.trajectories:
                raise NotLocalizedHalt(f'No pose available in trajectory for query at time {query_ts}.')

            ref_pose = kapture_data.trajectories[mapping_ts]
            query_pose = kapture_data.trajectories[query_ts]
            # compute relative pose
            # just use one cam of the pair as a comparison, take the first in line.
            sensor_id = next(iter(query_pose.keys()))
            query_from_world = query_pose[sensor_id]
            if sensor_id not in ref_pose:
                raise NotLocalizedHalt(f'no reference pose for {sensor_id} at time {mapping_ts}')
            refer_from_world = ref_pose[sensor_id]
            # The relative pose is from cam0 of the reference sequence to cam0 of the query sequence, respectively.
            # The 6DOF poses are specified as translation (t_x, t_y, t_z), and quaternion (q_x, q_y, q_z, q_w).
            refer_from_query = kapture.PoseTransform.compose([refer_from_world, query_from_world.inverse()])
            query_from_refer = refer_from_query.inverse()
            t_x, t_y, t_z = query_from_refer.t_raw
            q_w, q_x, q_y, q_z = query_from_refer.r_raw
            pose_entry = [mapping_ts, query_ts, t_x, t_y, t_z, q_x, q_y, q_z, q_w]  # note q_w was pushed at the end
            poses.append(pose_entry)

        except NotLocalizedHalt as e:
            logger.info('unable to localize because: ' + e)

        except Exception as e:
            logger.critical(e)
    logger.info(f'localized {len(poses)} ({len(poses) / len(pairs) * 100}%) poses.')
    logger.info(f'writing result to "{poses_file_path}"')
    with open(poses_file_path, 'wt') as f:
        for p in poses:
            f.write(' '.join(str(e) for e in p) + '\n')


def export_4seasons_pairs_command_line() -> None:
    """
    Exports 4seasons relative pose between the source and the query frame.
    Note that the relative pose is from cam0 of the reference sequence to cam0 of the query sequence, respectively.
    """
    parser = argparse.ArgumentParser(
        description='Exports 4seasons pair file.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i', '--input', required=True,
                        help='input path to kapture directory that contains both localized and mapping.')
    parser.add_argument('-p', '--pairs', required=True,
                        help='input path to timestamp pair file.')
    parser.add_argument('--no_heal', default=False, action='store_true',
                        help='do not overwrite timestamps from filenames.')
    parser.add_argument('-o', '--output', required=True,
                        help='output path to relative pose file.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    kapture_dir_path = path.abspath(args.input)
    pairs_file_path = path.abspath(args.pairs)
    poses_file_path = path.abspath(args.output)

    export_4seasons_pairfile(
        kapture_dir_path=kapture_dir_path,
        pairs_file_path=pairs_file_path,
        poses_file_path=poses_file_path,
        heal_timestamps=not args.no_heal)


if __name__ == '__main__':
    export_4seasons_pairs_command_line()
