# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Colmap specific database functions
"""

import logging
import numpy as np
import os
import os.path as path
from typing import List, Tuple, Dict, Optional, Set
from tqdm import tqdm

import sqlite3.dbapi2

# kapture
import kapture
from kapture.io.features import image_keypoints_from_file

# local
from .database import COLMAPDatabase, image_ids_to_pair_id, pair_id_to_image_ids
from .cameras import get_colmap_camera
from .export_colmap_reconstruction import export_to_colmap_txt

logger = logging.getLogger('colmap')


def foreign_keys_off(database: COLMAPDatabase) -> sqlite3.dbapi2.Cursor:
    """
    Set foreign keys off.
     Warning: don't forget to set it "on" afterwards!

    :param database: database to execute the command against
    :return: database cursor
    """
    return database.execute("PRAGMA foreign_keys = off;")


def foreign_keys_on(database: COLMAPDatabase) -> sqlite3.dbapi2.Cursor:
    """
    Set foreign keys on.

    :param database: database to execute the command against
    :return: database cursor
    """
    return database.execute("PRAGMA foreign_keys = on;")


def get_camera_ids_from_database(database: COLMAPDatabase) -> List[int]:
    """
    Get the list of colmap camera ids

    :param database: colmap database
    :return: list of colmap camera ids
    """
    return [camera_id for camera_id, in database.execute('SELECT camera_id FROM cameras;')]


def get_images_from_database(database: COLMAPDatabase) -> List[Tuple[str, int]]:
    """
    Get the list (image_name, camera_id) for the database

    :param database: colmap database
    :return: list of tuple (image_name, camera_id)
    """
    hide_progressbar = logger.getEffectiveLevel() > logging.INFO
    image_list = []
    for name, camera_id \
            in tqdm(database.execute('SELECT name, camera_id FROM images;'), disable=hide_progressbar):
        image_list.append((name, camera_id))
    return image_list


def get_keypoints_set_from_database(database: COLMAPDatabase,
                                    kapture_image_name_from_colmap_id: Dict[int, str]) -> Set[str]:
    """
    return set(image_name for image_name in colmap db keypoint table)

    :param database: colmap database
    :type database: COLMAPDatabase
    :param kapture_image_name_from_colmap_id: dict colmap image_id -> kapture image_name
    :type kapture_image_name_from_colmap_id: Dict[int, str]
    """
    return set(kapture_image_name_from_colmap_id[image_id]
               for (image_id,) in database.execute("SELECT image_id FROM keypoints"))


def get_matches_set_from_database(database: COLMAPDatabase,
                                  kapture_image_name_from_colmap_id: Dict[int, str]) -> Set[Tuple[str, str]]:
    """
    return set(pair_image_name for pair_image_name in colmap db matches table)

    :param database: colmap database
    :type database: COLMAPDatabase
    :param kapture_image_name_from_colmap_id: dict colmap image_id -> kapture image_name
    :type kapture_image_name_from_colmap_id: Dict[int, str]
    """
    colmap_matches = (pair_id_to_image_ids(pair_id)
                      for (pair_id,) in database.execute('SELECT pair_id FROM matches'))
    return set((kapture_image_name_from_colmap_id[pair[0]], kapture_image_name_from_colmap_id[pair[1]])
               for pair in colmap_matches)


def update_DB_cameras_and_poses(database: COLMAPDatabase,
                                kapture_data: kapture.Kapture) -> None:
    """
    Update the colmap database with cameras and pose from a kapture.

    :param database: input/output colmap database.
    :param kapture_data: kapture to read data from
    """
    if kapture_data.rigs:
        raise ValueError('update_DB_cameras_and_poses does not handle rigs, please use rigs_remove_inplace before.')

    # 1 - Remove foreign key constraints
    # 2 - empty camera table
    # 3 - insert new cameras with intrinsics
    # 4 - update images to use new camera IDs, and add pose from trajectories
    # 5 - Re-add foreign key constraints

    # WARNING: Delete all existing cameras
    logger.debug('Delete existing cameras in database...')
    foreign_keys_off(database)
    database.execute("DELETE from cameras")

    logger.debug('Add cameras in database...')
    colmap_camera_ids = {}  # to keep ID of camera in case it is different from the one we use
    for cam_id, cam in kapture_data.sensors.items():
        if cam.sensor_type != 'camera':
            continue
        col_cam_id, width, height, params, prior_focal_length = get_colmap_camera(cam)

        colmap_camera_ids[cam_id] = database.add_camera(col_cam_id,
                                                        # image size
                                                        width, height,
                                                        params,
                                                        prior_focal_length=prior_focal_length)

    logger.debug('Update images table with new camera ID and pose from trajectories')
    # compute actual camera trajectories from rig trajectories

    images_cam_ts = {image_path: (ts, cam_id)
                     for ts, shot in kapture_data.records_camera.items()
                     for cam_id, image_path in shot.items()}
    logger.info('register images in database...')
    for name, (timestamp, cam_id) in images_cam_ts.items():
        # retrieve image pose from trajectories
        if timestamp not in kapture_data.trajectories:
            # no pose for that timestamp # TODO what should we do ?
            prior_q = [0.0] * 4
            prior_t = [0.0] * 3
        else:
            assert cam_id in kapture_data.trajectories[timestamp]
            pose_tr = kapture_data.trajectories[timestamp].get(cam_id)
            prior_q = pose_tr.r_raw
            prior_t = pose_tr.t_raw

        # Update image in DB
        update_image(database, name, colmap_camera_ids[cam_id], prior_q=prior_q, prior_t=prior_t)

    # Foreign key constraints should be OK now
    foreign_keys_on(database)
    database.commit()


def remove_camera(database: COLMAPDatabase, camera_id: int) -> None:
    """
    Removes a camera from the colmap database (from cameras table only).

    :param database: input/output colmap database.
    :param camera_id: identifier of camera
    """
    try:
        database.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,))
    except Exception as e:
        logger.warning(e)
        pass


def update_image(database: COLMAPDatabase,
                 name,
                 camera_id,
                 prior_q=np.zeros(4),
                 prior_t=np.zeros(3)) -> None:
    """
    Update an image in the colmap database

    :param database: colmap database to update
    :param name: image name
    :param camera_id: camera identifier
    :param prior_q:
    :param prior_t:
    """
    try:
        database.execute(
            "UPDATE images SET camera_id = ?, prior_qw = ?, prior_qx = ?, prior_qy = ?, prior_qz = ?, prior_tx = ?,"
            " prior_ty = ?, prior_tz = ? WHERE name = ?",
            (camera_id, prior_q[0], prior_q[1], prior_q[2], prior_q[3], prior_t[0], prior_t[1], prior_t[2], name))
    except Exception as e:
        logger.warning(e)
        pass


def get_colmap_camera_ids_from_db(database: COLMAPDatabase, images: kapture.RecordsCamera) -> Dict[str, int]:
    """
    returns a dict mapping colmap_camera_ID -> sensor_id

    :param database: input colmap database.
    :param images: input kapture recorded images.
    :return: a dict of colmap camera ids kapture_id -> colmap_id
    """
    # colmap_camera_ID = {}  # colmap_camera_ID[sensor_id] = ids
    colmap_image_to_camera = {
        name: camera_id
        for camera_id, name in database.execute("SELECT camera_id, name FROM images")
    }
    colmap_camera_ids = {}
    for timestamp, sensor_id, sensing_filepath in kapture.flatten(images):
        if sensor_id not in colmap_camera_ids:
            colmap_camera_id = colmap_image_to_camera[sensing_filepath]
            colmap_camera_ids[sensor_id] = colmap_camera_id

    return colmap_camera_ids


def get_colmap_image_ids_from_db(database: COLMAPDatabase):
    """
    returns a dict mapping image_name -> colmap_image_ID

    :param database: input colmap database to write in.
    :return: a dict of image_name -> colmap_image_ID
    """
    # colmap_image_ids = {}  # colmap_image_ids[name] = ids
    return {
        name: image_id
        for image_id, name in database.execute("SELECT image_id, name FROM images")
    }


def is_colmap_db_empty(database: COLMAPDatabase) -> bool:
    """
    Check if the given colmap database is empty, ie. all colmap table are
        - are not there,
        - or empty

    :param database: input colmap database to write in.
    :return: True if the given colmap database is empty
    """
    # check database is empty
    table_names = ['cameras', 'descriptors', 'images', 'keypoints',
                   'matches', 'sqlite_sequence', 'two_view_geometries']
    for table in table_names:
        try:
            cursor = database.execute("SELECT count(*) FROM {}".format(table))
            count = cursor.fetchall()[0][0]
            if count > 0:
                return False
        except Exception:
            continue
    return True


def exists_table(table: str, database: COLMAPDatabase) -> bool:
    """
    Check if a table exist in a database

    :param table: name of table to check
    :param database: colmap database.
    :return: True if the given table exists
    """
    try:
        cursor = database.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{}'".format(table))
        count = cursor.fetchall()[0][0]
        if count == 0:
            return False
    except Exception:
        return False

    return True


def save_match_list(kapture_data: kapture.Kapture, path_file: str) -> None:
    """
    Saves matching files from db in a file at path_file
    format :
    image_match_1 image_match_2

    :param kapture_data: kapture data
    :param path_file: path to the matches file

    """
    if path.exists(path_file):
        os.remove(path_file)
    f = open(path_file, "w")
    for image_path1, image_path2 in kapture_data.matches:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()


def generate_priors_for_reconstruction(kapture_data: kapture.Kapture,
                                       database: COLMAPDatabase,
                                       path_to_priors_for_reconstruction: str) -> None:
    """
    Generate priors for a reconstruction using cameras, images and trajectories.

    :param kapture_data: kapture data
    :param database: colmap database.
    :param path_to_priors_for_reconstruction: path to the priors file
    """
    colmap_camera_ids = get_colmap_camera_ids_from_db(database, kapture_data.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(database)
    kapture_data_copy = kapture.Kapture(sensors=kapture_data.sensors,
                                        records_camera=kapture_data.records_camera,
                                        trajectories=kapture_data.trajectories,
                                        rigs=kapture_data.rigs)
    # in priors, do not copy keypoints, points3d
    export_to_colmap_txt(
        path_to_priors_for_reconstruction,
        kapture_data_copy,
        "",  # kapture_data_copy do not have binaries so path is irrelevant
        colmap_camera_ids,
        colmap_image_ids)


def add_cameras_to_database(sensors: kapture.Sensors, database: COLMAPDatabase) -> Dict[str, int]:
    """
    Add the kapture cameras to the colmap database.

    :param sensors: list af kapture cameras
    :param database: colmap database.
    :return: dictionary of kapture camera identifier -> colmap camera identifier
    """
    assert isinstance(sensors, kapture.Sensors)
    colmap_camera_ids = {}
    cameras = {cam_id: camera
               for cam_id, camera in sensors.items()
               if isinstance(camera, kapture.Camera)}

    for cam_id, cam in cameras.items():
        col_cam_id, width, height, params, prior_focal_length = get_colmap_camera(cam)
        colmap_camera_ids[cam_id] = database.add_camera(
            col_cam_id,
            width,
            height,
            params,
            prior_focal_length=prior_focal_length)

    database.commit()
    return colmap_camera_ids


def get_images_as_list_in_colmap_format(flatten_images: List[Tuple[int, str, str]],
                                        trajectories: Optional[kapture.Trajectories],
                                        colmap_camera_ids: Dict[str, int]
                                        ) -> List[Tuple[str, int, List[float], List[float]]]:
    """
    Returns data formatted to fill the colmap "images" table in the database.

    :param flatten_images: input list of images obtained through flatten(records_camera)
    :param trajectories: input optional kapture trajectories. Poses are set to 0 if not given.
    :param colmap_camera_ids: input dict mapping kapture sensor ids to colmap camera ids.
    :return: A list of tuple containing data to fill the colmap "images" table, ie
            image_filename, colmap_cam_id, rotation, translation
    """
    images_in_colmap_format = []
    for timestamp, sensor_id, image_filename in flatten_images:
        # retrieve image pose from trajectories
        prior_q, prior_t = None, None
        if trajectories is not None and (timestamp, sensor_id) in trajectories:
            # there is a pose for that timestamp / sensor_id pair
            pose_tr = trajectories[(timestamp, sensor_id)]
            prior_q = pose_tr.r_raw
            prior_t = pose_tr.t_raw
        # set fall back values if pose is undefined
        if prior_q is None:
            prior_q = 4 * [0.0]
        if prior_t is None:
            prior_t = 3 * [0.0]
        images_in_colmap_format.append(
            (image_filename, colmap_camera_ids[sensor_id], prior_q, prior_t)
        )
    # images_in_colmap_format = image_filename, colmap_cam_id, rotation, translation
    return images_in_colmap_format


def update_images_from_list_in_colmap_format(database: COLMAPDatabase,
                                             image_list: List[Tuple[str, str, List[float], List[float]]]) -> None:
    """
    Update images in the colmap database.

    :param database: colmap database.
    :param image_list: list of images information to update
    """
    for name, cam_id, prior_q, prior_t in image_list:
        update_image(database,
                     name, cam_id,
                     prior_q=prior_q,
                     prior_t=prior_t)
    database.commit()


def add_images_from_list_in_colmap_format(database: COLMAPDatabase,
                                          image_list: List[Tuple[str, str, List[float], List[float]]]
                                          ) -> Dict[str, int]:
    """
    Add images to the colmap database.

    :param database: input/output colmap database.
    :param image_list: list of image information
    :return: dict mapping kapture image ids to colmap image ids.
    """
    colmap_image_ids = {}  # colmap_image_ids[name] = ids
    for name, cam_id, prior_q, prior_t in image_list:
        colmap_image_ids[name] = database.add_image(
            name, cam_id,
            prior_q=prior_q,
            prior_t=prior_t)
    database.commit()
    return colmap_image_ids


def update_images_in_database_from_flatten(database: COLMAPDatabase,
                                           flatten_images: List[Tuple[int, str, str]],
                                           trajectories: kapture.Trajectories,
                                           colmap_camera_ids: dict) -> None:
    """
    Updates images information in colmap database.

    :param database: colmap database.
    :param flatten_images: list of image information
    :param trajectories: trajectories for the images
    :param colmap_camera_ids: kapture camera identifier -> colmap camera identifier dictionary
    """
    images_in_colmap_format = get_images_as_list_in_colmap_format(flatten_images, trajectories, colmap_camera_ids)
    update_images_from_list_in_colmap_format(database, images_in_colmap_format)


def add_images_to_database_from_flatten(database: COLMAPDatabase,
                                        flatten_images: List[Tuple[int, str, str]],
                                        trajectories: kapture.Trajectories,
                                        colmap_camera_ids: dict) -> Dict[str, int]:
    """
    Add images in the colmap database.

    :param database: colmap database.
    :param flatten_images: kapture images
    :param trajectories: images trajectories
    :param colmap_camera_ids: kapture camera identifier -> colmap camera identifier dictionary
    :return: dict mapping kapture image ids to colmap image ids.
    """
    images_in_colmap_format = get_images_as_list_in_colmap_format(flatten_images, trajectories, colmap_camera_ids)
    colmap_image_ids = add_images_from_list_in_colmap_format(database, images_in_colmap_format)
    return colmap_image_ids


def update_images_in_database(database: COLMAPDatabase,
                              images,
                              trajectories: kapture.Trajectories,
                              colmap_camera_ids: dict) -> None:
    """
    Update image information in colmap database.

    :param database: colmap database.
    :param images: kapture images
    :param trajectories: images trajectories
    :param colmap_camera_ids: kapture camera identifier -> colmap camera identifier dictionary
    """
    images_flattened = list(images.flattened())
    images_in_colmap_format = get_images_as_list_in_colmap_format(images_flattened, trajectories, colmap_camera_ids)
    update_images_from_list_in_colmap_format(database, images_in_colmap_format)


def add_images_to_database(database: COLMAPDatabase,
                           images: kapture.RecordsCamera,
                           trajectories: Optional[kapture.Trajectories],
                           colmap_camera_ids: Dict[str, int]
                           ) -> Dict[str, int]:
    """
    Add images in colmap database.

    :param database: colmap database.
    :param images: kapture images
    :param trajectories: images trajectories
    :param colmap_camera_ids: dict mapping kapture sensor ids to colmap camera ids.
    :return: dict mapping kapture image ids to colmap image ids.
    """
    assert isinstance(images, kapture.RecordsCamera)
    assert trajectories is None or isinstance(trajectories, kapture.Trajectories)
    images_flattened = list(kapture.flatten(images))
    images_in_colmap_format = get_images_as_list_in_colmap_format(images_flattened, trajectories, colmap_camera_ids)
    colmap_image_ids = add_images_from_list_in_colmap_format(database, images_in_colmap_format)
    return colmap_image_ids


def add_keypoints_to_database(database: COLMAPDatabase,
                              keypoints: kapture.Keypoints,
                              kapture_dir_path: str,
                              colmap_image_ids: dict) -> None:
    """
    Add keypoints to the colmap database.

    :param database: colmap database.
    :param keypoints: kapture keypoints to add
    :param kapture_dir_path: kapture data top directory
    :param colmap_image_ids: kapture camera identifier -> colmap camera identifier dictionary
    """
    keypoints_filepaths = kapture.io.features.keypoints_to_filepaths(keypoints, kapture_dir_path)
    for image_filename, keypoints_filepath in keypoints_filepaths.items():
        image_keypoints = image_keypoints_from_file(keypoints_filepath, keypoints.dtype, keypoints.dsize)
        colmap_image_id = colmap_image_ids[image_filename]
        # Make sure keypoints are np.float32 and support by colmap
        if image_keypoints.shape[1] not in {2, 4, 6}:
            image_keypoints = image_keypoints[:, 0:2]
        image_keypoints = image_keypoints.astype(np.float32)
        database.add_keypoints(colmap_image_id, image_keypoints)
    database.commit()


def add_descriptors_to_database(database: COLMAPDatabase,
                                descriptors: kapture.Descriptors,
                                kapture_dir_path: str,
                                colmap_image_ids: dict) -> None:
    """
    Add descriptors to the colmap database.

    :param database: colmap database.
    :param descriptors: kapture descriptors to add
    :param kapture_dir_path: kapture data top directory
    :param colmap_image_ids: kapture camera identifier -> colmap camera identifier dictionary
    """
    descriptors_filepaths = kapture.io.features.descriptors_to_filepaths(descriptors, kapture_dir_path)
    for image_filename, descriptors_filepath in descriptors_filepaths.items():
        image_descriptors = image_keypoints_from_file(descriptors_filepath, descriptors.dtype, descriptors.dsize)
        colmap_image_id = colmap_image_ids[image_filename]
        database.add_descriptors(colmap_image_id, image_descriptors)
    database.commit()


def add_matches_to_database(database: COLMAPDatabase,
                            matches: kapture.Matches,
                            kapture_dir_path: str,
                            colmap_image_ids: dict,
                            export_two_view_geometry: bool = False) -> None:
    """
    Add matches to the colmap database.

    :param database: colmap database.
    :param matches: kapture matches to add
    :param kapture_dir_path: kapture data top directory
    :param colmap_image_ids: kapture camera identifier -> colmap camera identifier dictionary
    :param export_two_view_geometry: if True, also export two geometry.
    """
    colmap_pairs_id = {}

    # matches[(image_path1, image_path2)] = image_matches
    matches.normalize()
    matches_filepaths = kapture.io.features.matches_to_filepaths(matches, kapture_dir_path)
    for (image_path1, image_path2), image_matches_filepath in matches_filepaths.items():
        image_matches = kapture.io.features.image_matches_from_file(image_matches_filepath)
        colmap_image_id1 = colmap_image_ids[image_path1]
        colmap_image_id2 = colmap_image_ids[image_path2]
        colmap_pair_id = image_ids_to_pair_id(colmap_image_id1, colmap_image_id2)
        if colmap_pair_id in colmap_pairs_id:
            logging.warning('{} IS ALREADY IN DATABASE ({}, {})'.format(
                colmap_pair_id, *colmap_pairs_id[colmap_pair_id]))
            continue
        colmap_pairs_id[colmap_pair_id] = (image_path1, image_path2)
        # convert kapture to colmap matches (drop the score col, and convert to int)
        image_matches = image_matches[:, :-1].astype(np.uint32)
        try:
            database.add_matches(colmap_image_id1, colmap_image_id2, image_matches)
            if export_two_view_geometry:
                database.add_two_view_geometry(colmap_image_id1, colmap_image_id2, image_matches)
        except Exception as err:
            logger.warning(f'({image_path1}, {image_path2}) failed: {err}')

    database.commit()


def kapture_to_colmap(kapture_data: kapture.Kapture,
                      kapture_dirpath: str,
                      database: COLMAPDatabase,
                      export_two_view_geometry: bool = False) -> None:
    """
    Export kapture data to colmap database.

    :param kapture_data: kapture data to export
    :param kapture_dirpath: path to kapture directory, to retrieve binary files (keypoints, descriptors, ...)
    :param database: colmap database.
    :param export_two_view_geometry: if True, also export two geometry.
    """
    assert isinstance(kapture_data, kapture.Kapture)
    assert kapture_data.sensors is not None
    assert kapture_data.records_camera is not None

    if not is_colmap_db_empty(database):
        raise ValueError('the existing colmap database is not empty : {}'.format(database))

    # compute actual camera trajectories from rigs trajectories
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

    database.create_tables()

    # cameras
    logger.info(f'registering {len(kapture_data.sensors)} sensors (cameras) in database...')
    colmap_camera_ids = add_cameras_to_database(kapture_data.sensors, database)

    # images
    logger.info(f'registering {len(list(kapture.flatten(kapture_data.records_camera)))} images in database...')
    colmap_image_ids = add_images_to_database(
        database, kapture_data.records_camera,
        kapture_data.trajectories, colmap_camera_ids)

    # keypoints
    if kapture_data.keypoints is not None:
        logger.info(f'registering {len(kapture_data.keypoints)} keypoints in database...')
        add_keypoints_to_database(database, kapture_data.keypoints, kapture_dirpath, colmap_image_ids)

    # descriptors
    if kapture_data.descriptors is not None:
        logger.info(f'registering {len(kapture_data.descriptors)} descriptors in database...')
        add_descriptors_to_database(database, kapture_data.descriptors, kapture_dirpath, colmap_image_ids)

    # matches
    if kapture_data.matches is not None:
        logger.info(f'registering {len(kapture_data.matches)} matches in database...')
        add_matches_to_database(database, kapture_data.matches, kapture_dirpath,
                                colmap_image_ids, export_two_view_geometry)
