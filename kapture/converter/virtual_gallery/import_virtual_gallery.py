# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Virtual gallery to kapture import functions.
"""

import os
import logging
from typing import List
# kapture
import kapture
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
# local
from .virtual_gallery_testing import import_testing_intrinsics, import_testing_extrinsics
from .virtual_gallery_testing import convert_testing_intrinsics, convert_testing_extrinsics
from .virtual_gallery_training import import_training_intrinsics, import_training_extrinsics
from .virtual_gallery_training import convert_training_intrinsics, convert_training_extrinsics
from .virtual_gallery_training import training_rig_config


logger = logging.getLogger('virtual_gallery')


def import_virtual_gallery(input_root_path: str,
                           configuration: str,
                           light_range: List[int],
                           loop_range: List[int],
                           camera_range: List[int],
                           occlusion_range: List[int],
                           as_rig: bool,
                           images_import_method: TransferAction,
                           kapture_path: str,
                           force_overwrite_existing: bool = False) -> None:
    """
    Creates a kapture with a virtual gallery.

    :param input_root_path: root path of virtual gallery
    :param configuration: training, testing or all (both)
    :param light_range: list of lights to include
    :param loop_range: list of training loops to include
    :param camera_range: list of training cameras to include
    :param occlusion_range: list of testing occlusion levels to include
    :param as_rig: in training trajectories, writes the position of the rig instead of individual cameras
    :param kapture_path: path to kapture top directory
    :param force_overwrite_existing: Silently overwrite kapture files if already exists.
    """
    # Check for existing files
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_overwrite_existing)

    offset = 0
    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()
    trajectories = kapture.Trajectories()
    rigs = kapture.Rigs()

    # Process all training data
    if configuration == "training" or configuration == "all":
        logger.info("Reading training files")
        camera_range_set = set(camera_range)
        training_intrinsics = import_training_intrinsics(input_root_path, light_range, loop_range, camera_range_set)
        training_extrinsics = import_training_extrinsics(input_root_path, light_range, loop_range, camera_range_set)

        convert_training_intrinsics(training_intrinsics, cameras)
        convert_training_extrinsics(offset, training_extrinsics, images, trajectories, as_rig)
        rigs.update(training_rig_config)

        offset += len(training_extrinsics)
    # Process all testing data
    if configuration == "testing" or configuration == "all":
        logger.info("Reading testing files")
        testing_intrinsics = import_testing_intrinsics(input_root_path, light_range, occlusion_range)
        testing_extrinsics = import_testing_extrinsics(input_root_path, light_range, occlusion_range)

        convert_testing_intrinsics(testing_intrinsics, cameras)
        convert_testing_extrinsics(offset, testing_extrinsics, images, trajectories)

        offset += len(testing_extrinsics)

    logger.info("Writing imported data to disk")
    kapture_data = kapture.Kapture(sensors=cameras, records_camera=images, trajectories=trajectories, rigs=rigs or None)
    # import images
    image_list = [name for _, _, name in kapture.flatten(kapture_data.records_camera)]
    import_record_data_from_dir_auto(input_root_path, kapture_path, image_list, images_import_method)
    kapture_to_dir(kapture_path, kapture_data)
