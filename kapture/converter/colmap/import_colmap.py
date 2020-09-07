# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
This script imports data from a COLMAP database and/or reconstruction files

 It will assume that if all rotation parameters are null, there is no prior rotation, and if all translation parameters
 are null, there is no prior translation (using null values instead of zeros)
"""

import logging
import os
import os.path as path
from typing import Optional, Set, Type, Union

# kapture
import kapture
import kapture.io.features
import kapture.io.records
import kapture.io.structure
import kapture.algo.merge_keep_ids
from kapture.io.records import TransferAction, import_record_data_from_dir_auto

# local
from .database import COLMAPDatabase
from .import_colmap_database import get_cameras_from_database, get_images_and_trajectories_from_database
from .import_colmap_database import get_keypoints_from_database, get_descriptors_from_database
from .import_colmap_database import get_matches_from_database
from .import_colmap_rigs import import_colmap_rig
from .import_colmap_reconstruction import import_from_colmap_cameras_txt, import_from_colmap_images_txt
from .import_colmap_reconstruction import import_from_colmap_points3d_txt

logger = logging.getLogger('colmap')


def import_colmap_database(colmap_database_filepath: str,
                           kapture_dir_path: str,
                           no_geometric_filtering: bool = False,
                           skip_reconstruction: bool = False,
                           keypoint_name: str = 'SIFT',
                           descriptor_name: str = 'SIFT') -> kapture.Kapture:
    """
    Converts colmap database file to kapture data.
    If kapture_dir_path is given, it creates keypoints, descriptors, matches files (if any).

    :param colmap_database_filepath: path to colmap database file.
    :param kapture_dir_path: path to kapture directory. Is used to store keypoints, descriptors and matches files.
                            If not given (None), is equivalent to skip_reconstruction == True.
    :param no_geometric_filtering:
    :param keypoint_name: name of the keypoints detector (by default, in colmap, its SIFT, but can be imported)
    :param descriptor_name: name of the keypoints descriptor (by default, in colmap, its SIFT, but can be imported)
    :param skip_reconstruction: skip the import of the kapture/reconstruction part,
                            ie. Keypoints, Descriptors, Matches.
    :return: kapture object
    """
    kapture_data = kapture.Kapture()

    logger.debug(f'loading colmap database {colmap_database_filepath}')
    db = COLMAPDatabase.connect(colmap_database_filepath)

    # Generate cameras
    logger.debug('parsing cameras in database.')
    kapture_data.sensors = get_cameras_from_database(db)

    # Images, Trajectories
    logger.debug('parsing images and trajectories in database.')
    kapture_data.records_camera, kapture_data.trajectories = get_images_and_trajectories_from_database(db)

    if kapture_dir_path is not None and not skip_reconstruction:
        os.makedirs(kapture_dir_path, exist_ok=True)

        # keypoints
        logger.debug('parsing keypoints in database...')
        kapture_data.keypoints = get_keypoints_from_database(
            db, kapture_data.records_camera, kapture_dir_path, keypoint_name)

        # descriptors
        logger.debug('parsing descriptors in database...')
        kapture_data.descriptors = get_descriptors_from_database(
            db, kapture_data.records_camera, kapture_dir_path, descriptor_name)

        # matches
        logger.debug('parsing matches in database...')
        kapture_data.matches = get_matches_from_database(
            db, kapture_data.records_camera, kapture_dir_path, no_geometric_filtering)

    db.close()
    return kapture_data


def import_colmap_from_reconstruction_files(reconstruction_dir_path: str,
                                            kapture_dir_path: Optional[str],
                                            skip: Set[Type[Union[kapture.Keypoints,
                                                                 kapture.Points3d,
                                                                 kapture.Observations]]]
                                            ) -> kapture.Kapture:
    """
    Converts colmap reconstruction files to kapture data.
    If kapture_dir_path is given, keypoints files are created, and potentially their observations.

    :param reconstruction_dir_path:
    :param kapture_dir_path: path to kapture directory. Is used to store keypoints files.
                            If not given (None), keypoints are automatically skipped.
    :param skip: can skip independently : Keypoints, Points3d or Observations.
                Note that Points3d and Observations are in the same file, so you should skip both to gain its reading.
    :return: kapture object
    """
    logger.debug(f'loading colmap reconstruction from:\n\t"{reconstruction_dir_path}"')
    if skip:
        logger.debug(f'loading colmap reconstruction skipping {", ".join(s.__name__ for s in skip)}')

    kapture_data = kapture.Kapture()
    reconstruction_file_paths = (path.join(reconstruction_dir_path, filename)
                                 for filename in ['cameras.txt', 'images.txt', 'points3D.txt'])
    colmap_cameras_filepath, colmap_images_filepath, colmap_points3d_filepath = reconstruction_file_paths

    proceed_keypoints = kapture.Keypoints not in skip and kapture_dir_path is not None
    proceed_points3d = kapture.Points3d not in skip and path.exists(colmap_points3d_filepath)
    proceed_observations = kapture.Observations not in skip and path.exists(colmap_points3d_filepath)

    if path.exists(colmap_cameras_filepath):
        logging.debug(f'parsing cameras from:\n\t"{path.basename(colmap_cameras_filepath)}"')
        kapture_data.sensors = import_from_colmap_cameras_txt(colmap_cameras_filepath)

    if path.exists(colmap_images_filepath):
        logging.debug(f'loading images from:\n\t"{path.basename(colmap_images_filepath)}"')
        kapture_dir_path_for_keypoints = kapture_dir_path if proceed_keypoints else None
        images, trajectories, keypoints = import_from_colmap_images_txt(
            colmap_images_filepath, kapture_dir_path_for_keypoints)

        kapture_data.records_camera = images
        kapture_data.trajectories = trajectories
        kapture_data.keypoints = keypoints

    if proceed_points3d or proceed_observations:
        assert kapture_data.records_camera is not None
        image_id_2_names = {ts: image_name
                            for ts, cam_id, image_name in kapture.flatten(kapture_data.records_camera, True)}
        logger.debug(f'parsing 3d points and observations from:\n\t"{path.basename(colmap_points3d_filepath)}"')
        points3d, observations = import_from_colmap_points3d_txt(colmap_points3d_filepath, image_id_2_names)
        kapture_data.points3d = points3d if proceed_points3d else None
        kapture_data.observations = observations if proceed_observations else None

    return kapture_data


def import_colmap(kapture_dir_path: Optional[str],  # noqa: C901: the import algorithm is well documented
                  colmap_database_filepath: str = None,
                  colmap_reconstruction_dir_path: str = None,
                  colmap_images_dir_path: str = None,
                  colmap_rig_filepath: str = None,
                  no_geometric_filtering: bool = False,
                  skip_reconstruction: bool = False,
                  force_overwrite_existing: bool = False,
                  images_import_strategy: TransferAction = TransferAction.link_absolute
                  ) -> kapture.Kapture:
    """
    Converts colmap files to kapture object.

    :param kapture_dir_path: path to kapture directory. Is used to store keypoints, descriptors and matches files.
                            If not given (None), keypoints, descriptors and matches are skipped.
    :param colmap_database_filepath: optional path to colmap database file.
    :param colmap_reconstruction_dir_path: optional path to colmap reconstruction directory.
    :param colmap_images_dir_path: directory path to colmap images. If given, a link to it will be created.
    :param colmap_rig_filepath: optional path to colmap rig file.
    :param no_geometric_filtering:
    :param skip_reconstruction: skip the import of the kapture/reconstruction part,
                                ie. Keypoints, Descriptors, Matches, Points3d, Observations.
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    :param images_import_strategy: input choice: how to copy image files.
    :return: kapture object
    """

    # sanity checks
    if kapture_dir_path and colmap_images_dir_path and images_import_strategy == TransferAction.skip:
        logger.warning('Images from colmap will not be copied (skip).')

    # prepare output directory
    if kapture_dir_path:
        kapture.io.structure.delete_existing_kapture_files(kapture_dir_path, force_erase=force_overwrite_existing)
        os.makedirs(kapture_dir_path, exist_ok=True)

    # 1: import database
    kapture_from_database = None
    if colmap_database_filepath:
        logger.debug(f'importing from database "{colmap_database_filepath}"')
        kapture_from_database = import_colmap_database(
            colmap_database_filepath, kapture_dir_path, no_geometric_filtering, skip_reconstruction)

    # 2: import reconstruction text files.
    kapture_data_reconstructed = None
    if colmap_reconstruction_dir_path:
        # do not overwrite keypoints files if any from database import
        what_to_skip_during_import_txt = set()
        # if keypoints already loaded from DB,
        # do not load them again and overwrite them
        # because keypoints from txt have less info:
        if kapture_from_database and kapture_from_database.keypoints:
            what_to_skip_during_import_txt.add(kapture.Keypoints)
        # skip_reconstruction=skip keypoints, Points3d, Observations
        if skip_reconstruction:
            what_to_skip_during_import_txt |= {kapture.Keypoints, kapture.Points3d, kapture.Observations}
        logger.debug(f'importing from reconstruction "{colmap_reconstruction_dir_path}"')
        kapture_data_reconstructed = import_colmap_from_reconstruction_files(
            colmap_reconstruction_dir_path, kapture_dir_path, what_to_skip_during_import_txt)

    # Merge data from database and reconstruction files
    if colmap_database_filepath and colmap_reconstruction_dir_path:
        # if both are present:
        # - kapture_data.sensors: merge both, with priority to reconstruction.
        # - kapture_data.trajectories: keep only reconstruction trajectories.
        # - kapture_data.observations: keep only reconstruction observations.
        # - kapture_data.points3d: only exists in colmap reconstruction
        # - kapture_data.*: anything else, keep only database

        # by default take all from database
        kapture_data = kapture_from_database
        # just replace trajectories, observations, points3d
        kapture_data.trajectories = kapture_data_reconstructed.trajectories
        kapture_data.observations = kapture_data_reconstructed.observations
        kapture_data.points3d = kapture_data_reconstructed.points3d
        # do a merge for sensors. If conflict prefer reconstruction:
        kapture_data.sensors.update(kapture_data_reconstructed.sensors)

    elif colmap_database_filepath:
        kapture_data = kapture_from_database
    elif colmap_reconstruction_dir_path:
        kapture_data = kapture_data_reconstructed
    else:
        raise ValueError('Neither database nor reconstruction files where given.')

    # if there is a rig ! lets restore it, and restore also timestamps
    if colmap_rig_filepath:
        rigs, records_camera, trajectories = import_colmap_rig(
            colmap_rig_filepath,
            kapture_data.records_camera,
            kapture_data.trajectories)

        kapture_data.rigs = rigs
        if records_camera:
            if not len(list(kapture.flatten(kapture_data.records_camera))) == len(
                    list(kapture.flatten(records_camera))):
                raise ValueError('inconsistent timestamp reconstruction in images')
            kapture_data.records_camera = records_camera

        if trajectories:
            if not len(list(kapture.flatten(kapture_data.trajectories))) == len(list(kapture.flatten(trajectories))):
                raise ValueError('inconsistent timestamp reconstruction in trajectories')
            kapture_data.trajectories = trajectories

    # finally import images
    if kapture_dir_path and colmap_images_dir_path and images_import_strategy != TransferAction.skip:
        filename_list = [f for _, _, f in kapture.flatten(kapture_data.records_camera)]
        logger.info(f'importing {len(filename_list)} image files ...')
        import_record_data_from_dir_auto(
            colmap_images_dir_path,
            kapture_dir_path,
            filename_list,
            images_import_strategy
        )

    return kapture_data
