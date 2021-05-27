# Copyright 2021-present NAVER Corp. Under BSD 3-clause license

"""
Upgrade operations.
"""

import os
import os.path as path
from typing import List, Optional
import shutil
# import numpy as np like in kapture.io.csv
# so that types written as "np.float32" are understood by read_old_image_features_csv
import numpy as np  # noqa: F401

import kapture
import kapture.utils.logging
import kapture.io.features
import kapture.io.csv
from kapture.utils.paths import populate_files_in_dirpath
from kapture.utils.logging import getLogger

CSV_FILENAMES_1_0 = [
    path.join('sensors', 'sensors.txt'),
    path.join('sensors', 'trajectories.txt'),
    path.join('sensors', 'rigs.txt'),
    path.join('sensors', 'records_camera.txt'),
    path.join('sensors', 'records_depth.txt'),
    path.join('sensors', 'records_lidar.txt'),
    path.join('sensors', 'records_wifi.txt'),
    path.join('sensors', 'records_bluetooth.txt'),
    path.join('sensors', 'records_gnss.txt'),
    path.join('sensors', 'records_accelerometer.txt'),
    path.join('sensors', 'records_gyroscope.txt'),
    path.join('sensors', 'records_magnetic.txt'),
    path.join('reconstruction', 'points3d.txt')]


def read_old_image_features_csv(csv_filepath: str):
    """
    Read the old image feature

    :param csv_filepath: the path to the csv file containing image features
    """
    with open(csv_filepath, 'r') as source_file:
        table = kapture.io.csv.table_from_file(source_file)
        line = list(table)[0]
        assert len(line) == 3
        name, dtype, dsize = line[0], line[1], int(line[2])

    # try to list all possible type from numpy that can be used in eval(dtype)
    from numpy import float, float32, float64, int32, uint8  # noqa: F401
    if isinstance(type(eval(dtype)), type):
        dtype = eval(dtype)
    else:
        raise ValueError('Expect data type ')
    return name, dtype, dsize


def upgrade_1_0_to_1_1_inplace(kapture_dirpath: str,  # noqa: C901: function a bit long but well documented
                               keypoints_type: Optional[str],
                               descriptors_type: Optional[str],
                               global_features_type: Optional[str],
                               descriptors_metric_type: str,
                               global_features_metric_type: str) -> None:
    """
    Do the upgrade from 1.0 to 1.1 version in place: will replace all the necessary files.

    """
    # some text files didn't change, just change their header
    for csv_filename in CSV_FILENAMES_1_0:
        csv_fullpath = path.join(kapture_dirpath, csv_filename)
        if path.isfile(csv_fullpath):
            getLogger().debug(f'converting {csv_fullpath}...')
            old_version = kapture.io.csv.get_version_from_csv_file(csv_fullpath)
            assert (old_version is None or old_version == '1.0')
            with open(csv_fullpath, 'r') as source_file:
                if old_version is not None:
                    source_file.readline()  # read and ignore header
                # write replacement header
                lines = source_file.read()
            with open(csv_fullpath, 'w') as source_file:
                lines_all = kapture.io.csv.KAPTURE_FORMAT_1 + kapture.io.csv.kapture_linesep + lines
                source_file.write(lines_all)
    # keypoints
    keypoints_dir_path = path.join(kapture_dirpath, 'reconstruction', 'keypoints')
    keypoints_csv_path = path.join(keypoints_dir_path, 'keypoints.txt')
    local_features_json_filepath = path.join(keypoints_dir_path, 'extract_local_features.json')
    if path.isdir(keypoints_dir_path) and path.isfile(keypoints_csv_path):
        getLogger().debug(f'converting {keypoints_dir_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(keypoints_csv_path)
        assert old_version is None or old_version == '1.0'
        name, dtype, dsize = read_old_image_features_csv(keypoints_csv_path)
        os.remove(keypoints_csv_path)
        if keypoints_type is None:
            assert name != ''
            keypoints_type = name
        keypoints = kapture.Keypoints(name, dtype, dsize)
        keypoints_csv_output_path = path.join(kapture_dirpath,
                                              kapture.io.csv.FEATURES_CSV_FILENAMES[kapture.Keypoints](keypoints_type))
        keypoints_output_dir = path.dirname(keypoints_csv_output_path)
        kapture.io.csv.keypoints_to_file(keypoints_csv_output_path, keypoints)

        # json file
        if path.isfile(local_features_json_filepath):
            local_features_json_output_file = path.join(keypoints_output_dir, 'extract_local_features.json')
            shutil.move(local_features_json_filepath, local_features_json_output_file)

        # now copy all .kpt files
        keypoints_filenames = list(populate_files_in_dirpath(keypoints_dir_path, '.kpt'))
        # cast to list before enumerating (or moved files will be listed multiple times)
        for keypoints_filename in keypoints_filenames:
            keypoints_output_file = path.join(keypoints_output_dir, keypoints_filename)
            keypoints_inpath = path.join(keypoints_dir_path, keypoints_filename)
            os.makedirs(path.dirname(keypoints_output_file), exist_ok=True)
            shutil.move(keypoints_inpath, keypoints_output_file)
            old_kp_dir = path.dirname(keypoints_inpath)
            if len(os.listdir(old_kp_dir)) == 0:
                os.removedirs(old_kp_dir)

    # descriptors
    descriptors_dir_path = path.join(kapture_dirpath, 'reconstruction', 'descriptors')
    descriptors_csv_path = path.join(descriptors_dir_path, 'descriptors.txt')
    if path.isdir(descriptors_dir_path) and path.isfile(descriptors_csv_path):
        getLogger().debug(f'converting {descriptors_dir_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(descriptors_csv_path)
        assert old_version is None or old_version == '1.0'
        assert keypoints_type is not None
        name, dtype, dsize = read_old_image_features_csv(descriptors_csv_path)
        os.remove(descriptors_csv_path)
        if descriptors_type is None:
            assert name != ''
            descriptors_type = name
        descriptors = kapture.Descriptors(name, dtype, dsize, keypoints_type, descriptors_metric_type)
        descriptors_csv_output_path = path.join(kapture_dirpath,
                                                kapture.io.csv.FEATURES_CSV_FILENAMES[kapture.Descriptors](
                                                    descriptors_type)
                                                )
        descriptors_output_dir = path.dirname(descriptors_csv_output_path)
        kapture.io.csv.descriptors_to_file(descriptors_csv_output_path, descriptors)
        # now copy all .desc files
        descriptors_filenames = list(populate_files_in_dirpath(descriptors_dir_path, '.desc'))
        # cast to list before enumerating (or moved files will be listed multiple times)
        for descriptors_filename in descriptors_filenames:
            descriptors_output_file = path.join(descriptors_output_dir, descriptors_filename)
            os.makedirs(path.dirname(descriptors_output_file), exist_ok=True)
            shutil.move(path.join(descriptors_dir_path, descriptors_filename), descriptors_output_file)
            old_desc_dir = path.dirname(path.join(descriptors_dir_path, descriptors_filename))
            if len(os.listdir(old_desc_dir)) == 0:
                os.removedirs(old_desc_dir)

    # matches
    matches_dir_path = path.join(kapture_dirpath, 'reconstruction', 'matches')
    matches_json_filepath = path.join(matches_dir_path, 'run_matching.json')
    if path.isdir(matches_dir_path):
        getLogger().debug(f'converting {matches_dir_path}...')
        assert keypoints_type is not None
        matches_output_dir = kapture.io.features.get_matches_fullpath(None, keypoints_type, kapture_dirpath)
        os.makedirs(matches_output_dir, exist_ok=True)

        # json file
        if path.isfile(matches_json_filepath):
            matches_json_output_file = path.join(matches_output_dir, 'run_matching.json')
            shutil.move(matches_json_filepath, matches_json_output_file)

        # now copy all .matches files
        matches_filenames = list(populate_files_in_dirpath(matches_dir_path, '.matches'))
        # cast to list before enumerating (or moved files will be listed multiple times)
        for matches_filename in matches_filenames:
            matches_output_file = path.join(matches_output_dir, matches_filename)
            os.makedirs(path.dirname(matches_output_file), exist_ok=True)
            shutil.move(path.join(matches_dir_path, matches_filename), matches_output_file)
            old_matches_dir = path.dirname(path.join(matches_dir_path, matches_filename))
            if len(os.listdir(old_matches_dir)) == 0:
                os.removedirs(old_matches_dir)

    # global features
    global_features_dir_path = path.join(kapture_dirpath, 'reconstruction', 'global_features')
    global_features_csv_path = path.join(global_features_dir_path, 'global_features.txt')
    global_features_json_filepath = path.join(global_features_dir_path, 'extract_global_features.json')
    if path.isdir(global_features_dir_path) and path.isfile(global_features_csv_path):
        getLogger().debug(f'converting {global_features_dir_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(global_features_csv_path)
        assert old_version is None or old_version == '1.0'
        assert keypoints_type is not None
        name, dtype, dsize = read_old_image_features_csv(global_features_csv_path)
        os.remove(global_features_csv_path)
        if global_features_type is None:
            assert name != ''
            global_features_type = name
        global_features = kapture.GlobalFeatures(name, dtype, dsize, global_features_metric_type)
        global_features_csv_output_path = path.join(kapture_dirpath,
                                                    kapture.io.csv.FEATURES_CSV_FILENAMES[kapture.GlobalFeatures](
                                                        global_features_type)
                                                    )
        global_features_output_dir = path.dirname(global_features_csv_output_path)
        kapture.io.csv.global_features_to_file(global_features_csv_output_path, global_features)

        # json file
        if path.isfile(global_features_json_filepath):
            global_features_json_output_file = path.join(global_features_output_dir, 'extract_global_features.json')
            shutil.move(global_features_json_filepath, global_features_json_output_file)

        # now copy all .gfeat files
        global_features_filenames = list(populate_files_in_dirpath(global_features_dir_path, '.gfeat'))
        # cast to list before enumerating (or moved files will be listed multiple times)
        for global_features_filename in global_features_filenames:
            global_features_output_file = path.join(global_features_output_dir, global_features_filename)
            os.makedirs(path.dirname(global_features_output_file), exist_ok=True)
            shutil.move(path.join(global_features_dir_path, global_features_filename), global_features_output_file)
            old_gfeat_dir = path.dirname(path.join(global_features_dir_path, global_features_filename))
            if len(os.listdir(old_gfeat_dir)) == 0:
                os.removedirs(old_gfeat_dir)

    # observations
    observations_csv_filename = path.join('reconstruction', 'observations.txt')
    observations_csv_path = path.join(kapture_dirpath, observations_csv_filename)
    if path.isfile(observations_csv_path):
        getLogger().debug(f'converting {observations_csv_path}...')
        old_version = kapture.io.csv.get_version_from_csv_file(observations_csv_path)
        assert old_version is None or old_version == '1.0'
        assert keypoints_type is not None
        observations = kapture.Observations()
        with open(observations_csv_path, 'r') as source_file:
            table = kapture.io.csv.table_from_file(source_file)
            # point3d_id, [image_path, feature_id]*
            for points3d_id_str, *pairs in table:
                points3d_id = int(points3d_id_str)
                if len(pairs) > 1:
                    image_paths = pairs[0::2]
                    keypoints_ids = pairs[1::2]
                    for image_path, keypoint_id in zip(image_paths, keypoints_ids):
                        observations.add(points3d_id, keypoints_type, image_path, int(keypoint_id))
        kapture.io.csv.observations_to_file(observations_csv_path, observations)
    getLogger().info('upgrade_1_0_to_1_1_inplace - all done!')


def upgrade_1_0_to_1_1_orphan_features(local_features_paths: List[str],
                                       global_features_paths: List[str]) -> None:
    """
    upgrade orphan features to kapture 1.1. Orphan features are features stored outside the kapture folder
    they must follow the kapture-localization recommendation
    https://github.com/naver/kapture-localization/blob/main/doc/tutorial.adoc#recommended-dataset-structure

    :param local_features_paths: examples dataset/local_features/r2d2 dataset/local_features/d2_tf
    :param global_features_paths: examples dataset/global_features/apgem dataset/global_features/delg
    """
    for local_features_path in local_features_paths:
        keypoints_path = os.path.join(local_features_path, 'keypoints')
        keypoints_type = kapture.io.features.guess_feature_name_from_path(keypoints_path)
        keypoints_csv_path = path.join(keypoints_path, 'keypoints.txt')
        if path.isfile(keypoints_csv_path):
            old_version = kapture.io.csv.get_version_from_csv_file(keypoints_csv_path)
            if old_version is not None and old_version != '1.0':
                getLogger().warning(f'{keypoints_path} not in version 1.0; skipped')
            else:
                getLogger().debug(f'upgrading {keypoints_path}')
                name, dtype, dsize = read_old_image_features_csv(keypoints_csv_path)
                keypoints = kapture.Keypoints(name, dtype, dsize)
                kapture.io.csv.keypoints_to_file(keypoints_csv_path, keypoints)

        descriptors_path = os.path.join(local_features_path, 'descriptors')
        descriptors_csv_path = path.join(descriptors_path, 'descriptors.txt')
        if path.isfile(descriptors_csv_path):
            old_version = kapture.io.csv.get_version_from_csv_file(descriptors_csv_path)
            if old_version is not None and old_version != '1.0':
                getLogger().warning(f'{descriptors_path} not in version 1.0; skipped')
            else:
                getLogger().debug(f'upgrading {descriptors_csv_path}')
                name, dtype, dsize = read_old_image_features_csv(descriptors_csv_path)
                descriptors = kapture.Descriptors(name, dtype, dsize, keypoints_type, 'L2')
                kapture.io.csv.descriptors_to_file(descriptors_csv_path, descriptors)

    for global_features_path in global_features_paths:
        # global_features_type = kapture.io.features.guess_feature_name_from_path(global_features_path)
        global_features_csv_path = path.join(global_features_path, 'global_features.txt')
        if not path.isfile(global_features_csv_path):
            global_features_csv_path = path.join(global_features_path, 'global_features', 'global_features.txt')
        if path.isfile(global_features_csv_path):
            old_version = kapture.io.csv.get_version_from_csv_file(global_features_csv_path)
            if old_version is not None and old_version != '1.0':
                getLogger().warning(f'{global_features_path} not in version 1.0; skipped')
                continue
            getLogger().debug(f'upgrading {global_features_csv_path}')
            name, dtype, dsize = read_old_image_features_csv(global_features_csv_path)
            global_features = kapture.GlobalFeatures(name, dtype, dsize, 'L2')
            kapture.io.csv.global_features_to_file(global_features_csv_path, global_features)
    getLogger().info('upgrade_1_0_to_1_1_orphan_features - all done!')
