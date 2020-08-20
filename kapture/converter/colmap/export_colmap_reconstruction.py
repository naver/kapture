# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Functions to export the reconstruction part of colmap (only text format is supported).
"""

import logging
import os.path as path
from typing import Dict, Tuple, List

# kapture
import kapture
import kapture.io.features
# local
from .cameras import CAMERA_MODEL_NAMES, get_colmap_camera

logger = logging.getLogger('colmap')


def export_to_colmap_cameras_txt(colmap_cameras_filepath: str,
                                 sensors: kapture.Sensors,
                                 colmap_camera_ids: Dict[str, int]) -> None:
    """
    Exports to colmap reconstruction file "cameras.txt".

    :param colmap_cameras_filepath: path to colmap file "cameras.txt" to be writen.
    :param sensors: sensors to be exported
    :param colmap_camera_ids: gives the correspondences between kapture camera id and colmap camera id
    """
    assert path.basename(colmap_cameras_filepath) == 'cameras.txt'
    assert isinstance(colmap_camera_ids, dict)
    assert isinstance(sensors, kapture.Sensors)
    cameras = {cam_id: cam
               for cam_id, cam in sensors.items()
               if isinstance(cam, kapture.Camera)}

    # cameras.txt
    cameras_colmap_header = '# Sensor list with one line of data per camera:\n' \
                            '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n' \
                            '# NB cameras : {}\n'.format(len(cameras))

    converted_cameras = [(cam_id, get_colmap_camera(cam)) for cam_id, cam in cameras.items()]
    with open(colmap_cameras_filepath, 'w') as fid:
        fid.write(cameras_colmap_header)
        lines = (' '.join([str(colmap_camera_ids[cam_id]), CAMERA_MODEL_NAMES[col_cam_id]] +
                          [str(int(width)), str(int(height))] +
                          [str(int(v)) if v.is_integer() else str(v) for v in params])
                 for cam_id, (col_cam_id, width, height, params, prior_focal_length) in converted_cameras
                 if cam_id in colmap_camera_ids)
        fid.write('\n'.join(lines))


def export_to_colmap_images_txt(colmap_images_filepath: str,
                                images: kapture.RecordsCamera,
                                trajectories: kapture.Trajectories,
                                colmap_camera_ids: Dict[str, int],
                                colmap_image_ids: Dict[str, int],
                                image_to_keypoints: Dict[str, List[Tuple[float, float, int]]]) -> None:
    """
    Exports kapture to colmap reconstruction file "images.txt".

    :param colmap_images_filepath: path to colmap file "images.txt" to be writen.
    :param images: images list to export
    :param trajectories: poses to export
    :param colmap_camera_ids: correspondences between kapture camera id and colmap camera id
    :param colmap_image_ids: correspondences between kapture image id (image path) and colmap image id
    :param image_to_keypoints: input image_filename -> [(x, y, point_id), (x, y, point_id), ...]
    """
    assert path.basename(colmap_images_filepath) == 'images.txt'
    assert isinstance(images, kapture.RecordsCamera)
    assert isinstance(trajectories, kapture.Trajectories)
    assert isinstance(colmap_camera_ids, dict)
    assert isinstance(colmap_image_ids, dict)
    assert isinstance(image_to_keypoints, dict)

    images_flattened = list(kapture.flatten(images))
    images_colmap_header = '# Image list with two lines of data per image:\n' \
                           '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n' \
                           '#   POINTS2D[] as (X, Y, POINT3D_ID)\n' \
                           '# NB IMAGES : {}\n'.format(len(images_flattened))

    with open(colmap_images_filepath, 'w') as fid:
        fid.write(images_colmap_header)
        for timestamp, sensor_id, sensing_filepath in images_flattened:
            colmap_cam_id = colmap_camera_ids[sensor_id]
            colmap_image_id = colmap_image_ids[sensing_filepath]
            # retrieve image pose from trajectories

            if timestamp not in trajectories:
                logging.debug('timestamp:{} not in trajectories'.format(timestamp))
                continue

            if sensor_id not in trajectories[timestamp]:
                logger.debug('camera {} not found in trajectories for timestamp {}.'.format(
                    sensor_id, timestamp))
                continue

            pose_tr = trajectories[timestamp].get(sensor_id)
            line = [colmap_image_id] + pose_tr.r_raw + pose_tr.t_raw + [colmap_cam_id, sensing_filepath]
            fid.write(' '.join('{}'.format(i) for i in line) + '\n')
            if sensing_filepath in image_to_keypoints:
                # POINTS2D[] as (X, Y, POINT3D_ID)
                p2d = [(str(x), str(y), str(point_id))
                       for x, y, point_id in image_to_keypoints[sensing_filepath]]
                fid.write(' '.join('{} {} {}'.format(i, j, k) for i, j, k in p2d))
            fid.write('\n')


def export_to_colmap_matches_txt(colmap_matches_filepath: str, matches: kapture.Matches) -> None:
    """
    Exports to colmap reconstruction file "matches.txt". This files can be used to redo the geometric filtering.

    :param colmap_matches_filepath: input path to colmap matches file to be writen.
    :param matches: input kapture.Matches
    """
    assert isinstance(matches, kapture.Matches)
    with open(colmap_matches_filepath, 'w') as fid:
        # matches[(image_path1, image_path2)] = image_matches
        for image_path1, image_path2 in matches:
            fid.write('{} {}\n'.format(image_path1, image_path2))


def export_to_colmap_points3d_txt(colmap_points3d_filepath: str,
                                  colmap_image_ids: Dict[str, int],
                                  points3d: kapture.Points3d = None,
                                  observations: kapture.Observations = None) -> None:
    """
    Exports to colmap points3d text file.

    :param colmap_points3d_filepath: path to colmap points3d file to be writen.
    :param colmap_image_ids: correspondences between kapture image id (image path) and colmap image id
    :param points3d: kapture points3d to export
    :param observations: kapture observations to export
    """
    assert isinstance(points3d, kapture.Points3d) or points3d is None
    assert isinstance(observations, kapture.Observations) or observations is None
    assert isinstance(colmap_image_ids, dict)
    points3d_colmap_header = '# 3D point list with one line of data per point:\n' \
                             '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
    with open(colmap_points3d_filepath, 'w') as fid:
        fid.write(points3d_colmap_header)
        if points3d:
            for i in range(points3d.shape[0]):
                point3d = points3d[i]
                line = '{} {} {} {} {} {} {} 0'.format(i,
                                                       point3d[0], point3d[1], point3d[2],
                                                       int(point3d[3]), int(point3d[4]), int(point3d[5]))
                if observations is not None and i in observations and len(observations[i]) > 0:
                    line += ' '
                    pairs = [(str(colmap_image_ids[name]), str(keypoint_id)) for name, keypoint_id in observations[i]]
                    line += ' '.join([str(s) for s in list(sum(pairs, ()))])
                line += '\n'
                fid.write(line)


def export_to_colmap_txt(colmap_reconstruction_dirpath: str,
                         kapture_data: kapture.Kapture,
                         kapture_dirpath: str,
                         colmap_camera_ids: Dict[str, int],
                         colmap_image_ids: Dict[str, int]) -> None:
    """
    Exports to colmap reconstruction text files.

    :param colmap_reconstruction_dirpath: path to directory where colmap reconstruction files will be stored.
    :param kapture_data: input kapture data
    :param kapture_dirpath: path to output directory, where colmap files will be stored
    :param colmap_camera_ids: gives the correspondences between kapture camera id and colmap camera id
    :param colmap_image_ids: gives the correspondences between kapture image id (image path) and colmap image id
    """
    assert isinstance(kapture_data, kapture.Kapture)
    assert isinstance(colmap_camera_ids, dict)
    assert isinstance(colmap_image_ids, dict)
    assert kapture_data.records_camera is not None
    assert kapture_data.sensors is not None

    if kapture_data.rigs:
        # check the rigs are not used in trajectories
        used_sensors = set(sensor_id
                           for shot in kapture_data.trajectories.values()
                           for sensor_id in shot.keys())
        rigs_ids = set(rig_id for rig_id in kapture_data.rigs.keys())
        if any(used_sensor in rigs_ids for used_sensor in used_sensors):
            raise ValueError(
                'colmap format does not handle rigs notation. '
                'Remove rig from trajectories beforehand (see rigs_remove_inplace)')

    # cameras.txt
    logger.info('creating colmap cameras.txt')
    export_to_colmap_cameras_txt(path.join(colmap_reconstruction_dirpath, 'cameras.txt'),
                                 kapture_data.sensors, colmap_camera_ids)

    # images.txt
    image_to_keypoints = {}
    if kapture_data.keypoints and kapture_data.points3d and kapture_data.observations:
        observations_reversed = {(image_filename, keypoint_idx): point3d_idx
                                 for point3d_idx, (image_filename, keypoint_idx) in
                                 kapture.flatten(kapture_data.observations)}

        # prepare images.txt even lines
        # POINTS2D[] as (X, Y, POINT3D_ID)
        keypoints_filepaths = kapture.io.features.keypoints_to_filepaths(kapture_data.keypoints, kapture_dirpath)
        for image_filename, image_keypoints_filepath in keypoints_filepaths.items():
            image_keypoints = kapture.io.features.image_keypoints_from_file(image_keypoints_filepath,
                                                                            kapture_data.keypoints.dtype,
                                                                            kapture_data.keypoints.dsize)
            image_to_keypoints[image_filename] = []
            for i in range(image_keypoints.shape[0]):
                point3d_idx = observations_reversed[(image_filename, i)] if (image_filename,
                                                                             i) in observations_reversed else -1
                image_to_keypoints[image_filename].append((image_keypoints[i, 0], image_keypoints[i, 1], point3d_idx))

    if kapture_data.records_camera is None or kapture_data.trajectories is None:
        logger.info('skipping colmap images.txt (missing images or trajectories).')
    else:
        logger.info('creating colmap images.txt')
        export_to_colmap_images_txt(path.join(colmap_reconstruction_dirpath, 'images.txt'),
                                    kapture_data.records_camera, kapture_data.trajectories,
                                    colmap_camera_ids,
                                    colmap_image_ids,
                                    image_to_keypoints)

    # image_matches.txt: to be imported as custom match (allow geometric verification)
    if kapture_data.matches:
        logger.info('creating image_matches.txt')
        export_to_colmap_matches_txt(path.join(colmap_reconstruction_dirpath, 'image_matches.txt'),
                                     kapture_data.matches)

    # points3D.txt
    logging.info('creating colmap points3D.txt')
    export_to_colmap_points3d_txt(path.join(colmap_reconstruction_dirpath, 'points3D.txt'), colmap_image_ids,
                                  kapture_data.points3d, kapture_data.observations)
