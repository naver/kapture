#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to export 4seasons relative pose between the source and the query frame.
Note that the relative pose is from cam0 of the reference sequence to cam0 of the query sequence, respectively.
see more details at:
https://github.com/pmwenzel/mlad-iccv2021#task
"""

import logging
import os.path as path
import numpy as np
import quaternion
from tqdm import tqdm

# kapture
import kapture
import kapture.io.csv as csv
import kapture.io.features
from kapture.utils.logging import getLogger
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

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


def export_4seasons_pairfile(input: str,
                             pairs: str,
                             output: str,
                             heal: bool = True,
                             force: bool = False,
                             verbose: Verbosity = DEFAULT_VERBOSITY,
                             quiet: bool = False):
    """
    Exports 4seasons relative pose between the source and the query frame.
    Note that the relative pose is from cam0 of the reference sequence to cam0 of the query sequence, respectively.

    :param input: input path to kapture directory that contains both localized and mapping.
    :param pairs:  input path to timestamp pair file.
    :param output: output path to relative pose file.
    :param heal: overwrite timestamps from filenames.
    :param force: accepted for compatibility, unused (output is always overwritten).
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    input, pairs, output = path.abspath(input), path.abspath(pairs), path.abspath(output)
    logger.info('loading kapture data ...')
    with csv.get_all_tar_handlers(input) as tar_handlers:
        kapture_data = csv.kapture_from_dir(input, tar_handlers=tar_handlers)

    if heal:
        logger.debug('recovering timestamps from filenames.')
        heal_timestamps_inplace(
            kapture_data.records_camera,
            kapture_data.trajectories)

    pairs = load_timestamp_pairs(pairs)
    logger.info('removing rigs')
    kapture.rigs_remove_inplace(kapture_data.trajectories, kapture_data.rigs)

    poses = []
    logger.info('Processing poses')
    hide_progress = logger.getEffectiveLevel() > logging.INFO
    for mapping_ts, query_ts in tqdm(pairs, disable=hide_progress):
        try:
            if mapping_ts not in kapture_data.trajectories:
                # mapping, means its probable
                raise IndexError(f'No pose available in trajectory for mapping at time {mapping_ts}. '
                                 f'Have you merged mapping and query in the same kapture dataset ?')

            ref_pose = kapture_data.trajectories[mapping_ts]

            if query_ts not in kapture_data.trajectories:
                logger.info(f'No pose available in trajectory for query at time {query_ts}.')
                # put dummy pose
                query_from_refer = kapture.PoseTransform()
            else:
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
    logger.info(f'writing result to "{output}"')
    with open(output, 'wt') as f:
        for p in poses:
            f.write(' '.join(str(e) for e in p) + '\n')


def main():
    run_cli(export_4seasons_pairfile, aliases={'--pairs': ['-p']})


if __name__ == '__main__':
    main()
