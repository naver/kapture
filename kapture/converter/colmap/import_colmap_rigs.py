# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Functions to import the rig part of colmap (json format).
"""

import json
import logging
from typing import Optional, Tuple
# kapture
import kapture
# local
from .cameras import get_camera_kapture_id_from_colmap_id


logger = logging.getLogger('colmap')


def import_colmap_rig_json(rigs_colmap: list,
                           images: Optional[kapture.RecordsCamera] = None,
                           trajectories: Optional[kapture.Trajectories] = None
                           ) -> Tuple[kapture.Rigs, kapture.RecordsCamera, Optional[kapture.Trajectories]]:
    """
    Build a kapture rig from colmap json file.

    :param rigs_colmap: colmap data describing the rig.
    :param images: input/output camera recordings: timestamps are modified to match
    :param trajectories: input/output trajectories: timestamps are modified to match
    :return: rigs, images and trajectories
    """
    rigs_kapture = kapture.Rigs()
    # camera_id (kapture) -> file prefix
    camera_prefixes = {}

    """ rigs_colmap
    [{
        "cameras": [
            {"camera_id": 1, "image_prefix": "leftraw/"},
            {"camera_id": 2, "image_prefix": "rightraw/"}
        ],
        "ref_camera_id": 1
    }]
    """

    for rig_idx_colmap, rig_colmap in enumerate(rigs_colmap):
        rig_id_kapture = f'rig{rig_idx_colmap}'  # make up a rig ID from its idx in colmap.
        for cam_colmap in rig_colmap['cameras']:
            # colmap_cam_id -> kapture_cam_id
            camera_id_colmap = cam_colmap['camera_id']
            camera_id_kapture = get_camera_kapture_id_from_colmap_id(camera_id_colmap)
            camera_prefixes[camera_id_kapture] = cam_colmap['image_prefix']
            # colmap does not store rig geometry, but only the fact there is one.
            pose_unknown = kapture.PoseTransform(r=None, t=None)
            rigs_kapture[rig_id_kapture, camera_id_kapture] = pose_unknown

    reconstructed_images = None
    reconstructed_trajectories = None
    if images:
        # image_filepath => (prefix, suffix)
        filepath_to_split_fix = {}
        # if there are images, modify their timestamps to match
        # first pass: gather actual timestamps from suffix
        # camera_suffixes = set()
        for timestamp, camera_id_kapture, image_filepath in kapture.flatten(images):
            if camera_id_kapture not in camera_prefixes:
                raise KeyError('unknown camera_id {}'.format(camera_id_kapture))
            camera_prefix = camera_prefixes[camera_id_kapture]
            if not image_filepath.startswith(camera_prefix):
                raise ValueError('inconsistent camera name')
            filepath_to_split_fix[image_filepath] = (
                image_filepath[0:len(camera_prefix)],
                image_filepath[len(camera_prefix):]
            )

        suffixes = sorted(set(suf for _, suf in filepath_to_split_fix.values()))
        suffix_to_timestamp = {suffix: idx for idx, suffix in enumerate(suffixes)}
        idx_to_timestamp = {
            colmap_idx: suffix_to_timestamp[filepath_to_split_fix[filepath][1]]
            for colmap_idx, _, filepath in kapture.flatten(images)
        }

        # second pass: reconstruct images with timestamp (frame number) instead of colmap idx
        reconstructed_images = kapture.RecordsCamera()
        for colmap_idx, camera_id_kapture, image_filepath in kapture.flatten(images):
            timestamp = idx_to_timestamp[colmap_idx]
            reconstructed_images[timestamp, camera_id_kapture] = image_filepath

        # third pass: [optional] reconstruct trajectories
        if trajectories:
            reconstructed_trajectories = kapture.Trajectories()
            for colmap_idx, camera_id_kapture, pose in kapture.flatten(trajectories):
                timestamp = idx_to_timestamp[colmap_idx]
                reconstructed_trajectories[timestamp, camera_id_kapture] = pose

    return rigs_kapture, reconstructed_images, reconstructed_trajectories


def import_colmap_rig(json_filepath: str,
                      images: kapture.RecordsCamera = None,
                      trajectories: kapture.Trajectories = None):
    """
    Build a kapture rig from colmap json file.

    :param json_filepath: colmap reconstruction json file path
    :param images: input/output camera recordings: timestamps are modified to match
    :param trajectories: input/output trajectories: timestamps are modified to match
    :return: rigs, images and trajectories
    """
    with open(json_filepath) as infile:
        rigs_colmap = json.load(infile)
    return import_colmap_rig_json(rigs_colmap, images, trajectories)
