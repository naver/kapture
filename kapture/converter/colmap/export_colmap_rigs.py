# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Functions to export the rig part of colmap in json format.
"""

import json
import logging
import os.path as path
from typing import Dict, List

# kapture
import kapture

logger = logging.getLogger('colmap')


def export_colmap_rig_json(kapture_rigs: kapture.Rigs,
                           records_camera: kapture.RecordsCamera,
                           colmap_camera_ids: Dict[str, int]
                           ) -> List[Dict[str, list]]:
    """
    From colmap source code comments (colmap.cc:1401) :

    // Read the configuration of the camera rigs from a JSON file. The input images
    // of a camera rig must be named consistently to assign them to the appropriate
    // camera rig and the respective kapture image.
    //
    // An example configuration of a single camera rig:
    // [
    //   {
    //     "ref_camera_id": 1,
    //     "cameras":
    //     [
    //       {
    //           "camera_id": 1,
    //           "image_prefix": "left1_image"
    //       }, ...
    //     ]
    //   }
    // ]


    // This file specifies the configuration for a single camera rig and that you
    // could potentially define multiple camera rigs.


    Images with the same prefix ("left1_image/") are assigned to the same camera in the rig.
    Images with the same suffix ("_frame000.png" and "/frame000.png") are
    assigned to the same camera record, i.e., they are assumed to be captured at the same time.

    This code, try to retrieve prefix of images.

    :param kapture_rigs: kapture rigs to export
    :param records_camera: used to guess camera prefix: assume a directory per camera
    :param colmap_camera_ids: dict mapping sensor_id -> colmap_camera_ID
    :return: The structure as expected in the colmap json file.
    """

    colmap_rigs = []
    for rig_kapture in kapture_rigs.values():
        # rig[sensor_device_id] = <Pose>
        rig_colmap = {'cameras': []}
        for camera_id in rig_kapture:
            if camera_id not in colmap_camera_ids:
                # e.g. lidar0 and other sensors without record on disk
                logger.warning(f'Camera {camera_id} not in COLMAP cameras')
                continue
            colmap_camera_id = colmap_camera_ids[camera_id]
            # recover the image prefix :
            images_dir_paths = set(path.dirname(record_filepath)
                                   for _, sensor_id, record_filepath in kapture.flatten(records_camera)
                                   if sensor_id == camera_id)
            # check there is one, and only one image prefix (required by colmap).
            if not len(images_dir_paths) == 1:
                raise ValueError(f'unable to find an image for camera {camera_id}')

            the_images_dirpath = next(iter(images_dir_paths))
            if not the_images_dirpath:
                raise ValueError(f'unable to find prefix for images {the_images_dirpath}.')
            rig_colmap['cameras'].append({
                'camera_id': colmap_camera_id,
                'image_prefix': the_images_dirpath
            })
        rig_colmap['ref_camera_id'] = rig_colmap['cameras'][0]['camera_id']
        colmap_rigs.append(rig_colmap)

    return colmap_rigs


def export_colmap_rig(colmap_rig_filepath: str,
                      rigs: kapture.Rigs,
                      records_camera: kapture.RecordsCamera,
                      colmap_camera_ids: dict) -> None:
    """
    Exports kapture rigs to colmap rigs file.

    :param colmap_rig_filepath: file path where to write rig.
    :param rigs: kapture rigs.
    :param records_camera: used to guess camera prefix: assume a directory per camera
    :param colmap_camera_ids: dict mapping colmap_camera_ID -> sensor_id
    """
    colmap_rigs = export_colmap_rig_json(rigs, records_camera, colmap_camera_ids)
    with open(colmap_rig_filepath, 'w') as outfile:
        json.dump(colmap_rigs, outfile, indent=4, sort_keys=True)
