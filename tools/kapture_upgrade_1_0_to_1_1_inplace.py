#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data in version 1.0 to version 1.1 inplace
"""

import argparse
import logging
# import numpy as np like in kapture.io.csv
# so that types written as "np.float32" are understood by read_old_image_features_csv
import numpy as np  # noqa: F401

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
import kapture.io.features
import kapture.io.csv
from kapture.utils.upgrade import upgrade_1_0_to_1_1_inplace

logger = logging.getLogger('upgrade_1_0_to_1_1_inplace')


def upgrade_1_0_to_1_1_inplace_command_line() -> None:
    """
    Convert kapture data in version 1.0 to version 1.1 inplace.
    """
    parser = argparse.ArgumentParser(
        description='convert kapture data in version 1.0 to version 1.1')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    # export ###########################################################################################################
    parser.add_argument('-i', '--input', required=True, help='input path to kapture directory')
    parser.add_argument('--keypoints-type', default=None,
                        help='types of keypoints.')
    parser.add_argument('--descriptors-type', default=None, help='types of descriptors.')
    parser.add_argument('--descriptors-metric-type', default='L2', help='types of descriptors.')
    parser.add_argument('--global-features-type', default=None,
                        help='types of global features.')
    parser.add_argument('--global-features-metric-type', default='L2', help='types of descriptors.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    upgrade_1_0_to_1_1_inplace(args.input,
                               args.keypoints_type, args.descriptors_type, args.global_features_type,
                               args.descriptors_metric_type, args.global_features_metric_type)


if __name__ == '__main__':
    upgrade_1_0_to_1_1_inplace_command_line()
