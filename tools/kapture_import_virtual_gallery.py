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
import argparse
# kapture
import path_to_kapture  # noqa: F401
import kapture.utils.logging
from kapture.converter.virtual_gallery.import_virtual_gallery import import_virtual_gallery
from kapture.io.records import TransferAction

logger = logging.getLogger('virtual_gallery')


def import_virtual_gallery_command_line() -> None:
    """
    Do the virtual galley to kapture import using the command line parameters provided by the user.
    """
    parser = argparse.ArgumentParser(description='import virtual_gallery to kapture format')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)

    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output if already exists.')
    # import ###########################################################################################################
    parser.add_argument('-i', '--input', required=True, help=('input path to virtual_gallery '
                                                              'v10 root directory'))
    parser.add_argument('-o', '--output', required=True, help='output directory')
    parser.add_argument('-c', '--configuration', default="all", choices=["training", "testing", "all"],
                        help='training, testing or all (both)')
    parser.add_argument('--light-range', nargs='+', default=list(range(1, 7)), type=int,
                        help='list of lights to include')
    parser.add_argument('--loop-range', nargs='+', default=list(range(1, 6)), type=int,
                        help='list of training loops to include')
    parser.add_argument('--camera-range', nargs='+', default=list(range(0, 6)), type=int,
                        help='list of training cameras to include')
    parser.add_argument('--occlusion-range', nargs='+', default=list(range(1, 5)), type=int,
                        help='list of testing occlusion levels to include')
    parser.add_argument('--as-rig', action='store_true', default=False,
                        help='in training trajectories, writes the position of the rig instead of individual cameras')
    parser.add_argument('--image_transfer', type=TransferAction, default=TransferAction.skip,
                        help=f'How to import images [skip], '
                        f'choose among: {", ".join(a.name for a in TransferAction)}')
    ####################################################################################################################

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    import_virtual_gallery(
        args.input,
        args.configuration,
        args.light_range,
        args.loop_range,
        args.camera_range,
        args.occlusion_range,
        args.as_rig,
        args.image_transfer,
        args.output,
        args.force
    )


if __name__ == '__main__':
    import_virtual_gallery_command_line()
