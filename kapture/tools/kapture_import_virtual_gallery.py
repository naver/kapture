#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Script to import a virtual gallery to the kapture format.
Virtual Gallery is a dataset produced by NaverLabs Europe.
It is available at https://europe.naverlabs.com/research/3d-vision/virtual-gallery-dataset/
It is subject to the Creative Commons Attribution Non Commercial No Derivatives 4.0 International License.

    Virtual gallery v10 look like this:
    v10 # input_root
    ├── training
        └── gallery_lightX_loopY
            └── extrinsic.txt # extrinsics for each image
            └── intrinsic.txt # intrinsics for each image, always the same
            └── frames
                 └── rgb
                     └── camera_Z
                            └── rgb_00229.jpg  # images
    ├── testing/
        └── gallery_lightX_occlusionW
            └── extrinsic.txt # extrinsics for each image
            └── intrinsic.txt # intrinsics for each image, always the same
            └── frames
                 └── rgb
                     └── camera_0
                            └── rgb_00229.jpg  # images
"""

import logging
from typing import List, Literal
# kapture
import kapture.converter.virtual_gallery.import_virtual_gallery as virtual_gallery_import
from kapture.io.records import TransferAction
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('virtual_gallery')


def import_virtual_gallery(
        input: str,
        output: str,
        configuration: Literal['training', 'testing', 'all'] = 'all',
        light_range: List[int] = list(range(1, 7)),
        loop_range: List[int] = list(range(1, 6)),
        camera_range: List[int] = list(range(0, 6)),
        occlusion_range: List[int] = list(range(1, 5)),
        as_rig: bool = False,
        image_transfer: TransferAction = TransferAction.skip,
        force: bool = False,
        verbose: Verbosity = DEFAULT_VERBOSITY,
        quiet: bool = False
) -> None:
    """
    Imports virtual_gallery to kapture format.

    :param input: path to virtual_gallery v10 root directory
    :param output: output directory
    :param configuration: training, testing or all (both)
    :param light_range: list of lights to include
    :param loop_range: list of training loops to include
    :param camera_range: list of training cameras to include
    :param occlusion_range: list of testing occlusion levels to include
    :param as_rig: in training trajectories, writes the position of the rig instead of individual cameras
    :param image_transfer: how to import images
    :param force: delete output if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    virtual_gallery_import.import_virtual_gallery(
        input,
        configuration,
        light_range,
        loop_range,
        camera_range,
        occlusion_range,
        as_rig,
        image_transfer,
        output,
        force
    )


def main():
    run_cli(import_virtual_gallery, aliases={'--configuration': ['-c']})


if __name__ == '__main__':
    main()
