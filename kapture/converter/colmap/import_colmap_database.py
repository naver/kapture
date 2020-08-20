# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Colmap database import as basic kapture objects functions
"""

import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional

# kapture
import kapture
import kapture.io.features
# local
from .database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
from .database_extra import exists_table
from .cameras import CAMERA_MODEL_NAMES, get_camera_kapture_id_from_colmap_id

logger = logging.getLogger('colmap')


def get_cameras_from_database(database: COLMAPDatabase) -> kapture.Sensors:
    """
    Creates kapture sensors from the colmap database.

    :param database: colmap database
    :return: kapture sensors
    """
    logger.info('parsing cameras  ...')
    kapture_cameras = kapture.Sensors()

    for camera_id, model_id, width, height, params, prior_focal_length in database.execute(
            'SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras;'):
        if model_id not in CAMERA_MODEL_NAMES:
            logger.warning(f'unable to convert colmap camera model ({model_id}) for camera {camera_id}.')
            # use 0 as default
            model_id = 0

        camera_id = get_camera_kapture_id_from_colmap_id(camera_id)
        model_name = CAMERA_MODEL_NAMES[model_id]

        #  By setting the prior_focal_length flag to 0 or 1,
        #  you can give a hint whether the reconstruction algorithm should trust the focal length value.
        params = blob_to_array(params, np.float64)
        params = [width, height] + params.tolist()

        kapture_camera = kapture.Camera(model_name, params)
        kapture_cameras[camera_id] = kapture_camera
    return kapture_cameras


def get_images_and_trajectories_from_database(database: COLMAPDatabase
                                              ) -> Tuple[kapture.RecordsCamera, kapture.Trajectories]:
    """
    Creates records_camera and trajectories from colmap images table
    In trajectories, timestamps are made up from colmap image id.

    :param database: colmap database
    :return: kapture records_camera and trajectories
    """
    logging.info('parsing images ...')
    kapture_images = kapture.RecordsCamera()
    kapture_trajectories = kapture.Trajectories()
    hide_progressbar = logger.getEffectiveLevel() > logging.INFO
    for image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz \
            in tqdm(database.execute('SELECT image_id, name, camera_id, '
                                     'prior_qw, prior_qx, prior_qy, prior_qz, '
                                     'prior_tx, prior_ty, prior_tz  FROM images;'
                                     ), disable=hide_progressbar):
        # images
        timestamp = int(image_id)
        camera_id = get_camera_kapture_id_from_colmap_id(camera_id)
        kapture_images[timestamp, camera_id] = name
        # trajectories
        prior_q = [prior_qw, prior_qx, prior_qy, prior_qz]
        prior_t = [prior_tx, prior_ty, prior_tz]
        # do not register the pose part if its invalid.
        is_undefined = all(v is None for v in prior_q + prior_t)
        if is_undefined:
            # just ignore
            continue
        prior_pose = kapture.PoseTransform(prior_q, prior_t)
        kapture_trajectories[timestamp, camera_id] = prior_pose

    if len(kapture_trajectories) == 0:
        # if there is no pose at all, just don't bother.
        kapture_trajectories = None

    return kapture_images, kapture_trajectories


def get_keypoints_from_database(database: COLMAPDatabase,
                                records_camera: kapture.RecordsCamera,
                                kapture_dirpath: str,
                                keypoint_name: str = 'SIFT'
                                ) -> Optional[kapture.Keypoints]:
    """
    Writes keypoints files and return the kapture keypoints from the colmap database.
    Requires records_camera timestamp == colmap_image_id

    :param database: colmap database.
    :param records_camera: input images.
    :param kapture_dirpath: input root path to kapture.
    :param keypoint_name: name of the keypoints detector (by default, in colmap, its SIFT, but can be imported)
    :return: kapture keypoints
    """
    image_filenames = set()
    dtype = np.float32
    dsize = None  # usually 6, will be retrieved on first keypoints of DB
    # DB query
    colmap_keypoints_request = (
        (image_id, blob_to_array(data, dtype, (rows, cols)) if (rows > 0 and cols > 0) else np.zeros((0, 6)))
        for image_id, rows, cols, data in database.execute("SELECT image_id, rows, cols, data FROM keypoints"))

    hide_progressbar = logger.getEffectiveLevel() > logging.INFO
    for colmap_image_id, image_keypoints in tqdm(colmap_keypoints_request, disable=hide_progressbar):
        if dsize is None:
            assert image_keypoints.dtype == dtype
            dsize = int(image_keypoints.shape[1])
        elif dsize != image_keypoints.shape[1]:
            raise ValueError('inconsistent keypoints size or type.')
        # retrieve image path from image_id
        assert len(records_camera[colmap_image_id]) == 1
        timestamp = colmap_image_id
        image_filename = next((v for v in records_camera[timestamp].values()), None)
        assert image_filename
        keypoints_filepath = kapture.io.features.get_keypoints_fullpath(kapture_dirpath, image_filename)
        if image_keypoints.shape[0] == 0:
            logger.warning(f'image={image_filename} has 0 keypoints')
        # save the actual file
        kapture.io.features.image_keypoints_to_file(keypoints_filepath, image_keypoints)
        # register it into kapture
        image_filenames.add(image_filename)

    if image_filenames:
        return kapture.Keypoints(keypoint_name, dtype, dsize, image_filenames)
    else:
        return None


def get_descriptors_from_database(database: COLMAPDatabase,
                                  images: kapture.RecordsCamera,
                                  kapture_dirpath: str,
                                  descriptor_name: str = 'SIFT'
                                  ) -> Optional[kapture.Descriptors]:
    """
    Writes descriptors files and return the list in kapture format from the colmap database.

    :param database: colmap database.
    :param images: list of images (as RecordsCamera).
    :param kapture_dirpath: input root path to kapture.
    :param descriptor_name: name of the keypoints descriptor (by default, in colmap, its SIFT, but can be imported)
    :return: kapture descriptors
    """
    image_filenames = set()
    dtype = np.uint8  # values in the range 0…255
    # see https://colmap.github.io/tutorial.html#feature-detection-and-extraction
    dsize = None  # usually uint8, 128, will be retrieved on first descriptor of DB
    colmap_descriptors = (
        (image_id, blob_to_array(data, dtype, (rows, cols)) if (rows > 0 and cols > 0) else np.zeros((0, dsize)))
        for image_id, rows, cols, data in database.execute("SELECT image_id, rows, cols, data FROM descriptors"))
    hide_progressbar = logger.getEffectiveLevel() > logging.INFO
    for image_id, image_descriptors in tqdm(colmap_descriptors, disable=hide_progressbar):
        # retrieve image path from image_id (actually the timestamp)
        if dsize is None:
            dsize = int(image_descriptors.shape[1])
        elif dsize != image_descriptors.shape[1] or dtype != image_descriptors.dtype:
            raise ValueError('inconsistent descriptors size or type.')

        image_filename = next((v for v in images[image_id].values()), None)
        assert image_filename
        descriptors_filepath = kapture.io.features.get_descriptors_fullpath(kapture_dirpath, image_filename)
        if image_descriptors.shape[0] == 0:
            logger.warning(f'image={image_id}:{image_filename} has 0 descriptors.')
        # save the actual file
        kapture.io.features.image_keypoints_to_file(descriptors_filepath, image_descriptors)
        # register it into to kapture
        image_filenames.add(image_filename)

    if image_filenames:
        return kapture.Descriptors(descriptor_name, dtype, dsize, image_filenames)
    else:
        return None


def get_matches_from_database(database: COLMAPDatabase,
                              images: kapture.RecordsCamera,
                              kapture_dirpath: str,
                              no_geometric_filtering: bool) -> kapture.Matches:
    """
    Writes Matches files and return the list in kapture format from the colmap database.

    :param database: input colmap database.
    :param images: input list of images (as RecordsCamera).
    :param kapture_dirpath: input root path to kapture.
    :param no_geometric_filtering: only retrieve matches with geometric consistency.
    :return: kapture matches
    """
    kapture_matches = kapture.Matches()
    # check there is geometric matches available
    matches_table_name = 'matches'
    if not no_geometric_filtering:
        if not exists_table('two_view_geometries', database):
            logger.warning("No table: two_view_geometries: skipping geometric filtering")
        else:
            request = database.execute(
                'SELECT COUNT (*) FROM two_view_geometries')
            nb_verified_matches = next(request)[0]
            if nb_verified_matches > 0:
                request = database.execute('SELECT COUNT (*) FROM matches')
                nb_total_matches = next(request)[0]
                logger.info('keeps {}% of verified matches ({}/{}) ...'.format(
                    nb_verified_matches / nb_total_matches * 100, nb_verified_matches, nb_total_matches))
                matches_table_name = 'two_view_geometries'

    colmap_matches = [(pair_id_to_image_ids(pair_id), blob_to_array(data, np.uint32, (rows, cols)))
                      if rows > 0 else (pair_id_to_image_ids(pair_id), np.empty((rows, cols), dtype=np.uint32))
                      for pair_id, rows, cols, data
                      in database.execute('SELECT pair_id, rows, cols, data FROM {}'.format(matches_table_name))]

    logger.debug('matches: {}'.format(len(colmap_matches)))
    hide_progressbar = logger.getEffectiveLevel() > logging.INFO
    for (image_id1, image_id2), image_matches in tqdm(colmap_matches, disable=hide_progressbar):
        if image_id1 not in images or image_id2 not in images:
            logger.critical('inconsistent image ID {} or {}'.format(image_id1, image_id2))
            continue
        filename1 = next((v for v in images[image_id1].values()), None)
        filename2 = next((v for v in images[image_id2].values()), None)
        assert filename1 and filename2
        if (filename1, filename2) != kapture.Matches.lexical_order(filename1, filename2):
            # have to swap matches (keypoint image1, keypoint image2) become (keypoint image2, keypoint image1)
            image_matches = image_matches[:, ::-1]
            filename1, filename2 = kapture.Matches.lexical_order(filename1, filename2)

        # actually write the file
        # convert colmap image matches into kapture (cast to float and add a score column)
        image_matches = image_matches.astype(np.float)
        image_matches = np.hstack([image_matches, np.zeros((image_matches.shape[0], 1))])
        image_matches_filepath = kapture.io.features.get_matches_fullpath((filename1, filename2), kapture_dirpath)
        kapture.io.features.image_matches_to_file(image_matches_filepath, image_matches)
        # register the matching in kapture
        kapture_matches.add(filename1, filename2)

    return kapture_matches
