#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data in version 1.0 to version 1.1 (only orphan features)
"""

import logging
import argparse

import path_to_kapture  # noqa: F401
import kapture.utils.logging
from kapture.utils.upgrade import upgrade_1_0_to_1_1_orphan_features

logger = logging.getLogger('upgrade_1_0_to_1_1_orphan_features')


def upgrade_1_0_to_1_1_orphan_features_command_line() -> None:
    """
    upgrade orphan features to kapture 1.1. Orphan features are features stored outside the kapture folder
    they must follow the kapture-localization recommendation
    https://github.com/naver/kapture-localization/blob/main/doc/tutorial.adoc#recommended-dataset-structure
    """
    parser = argparse.ArgumentParser(
        description='convert kapture data in version 1.0 to version 1.1 (only orphan features)')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    # export ###########################################################################################################
    parser.add_argument('-gfeat', '--global-features', nargs='+', default=[],
                        help=('paths to the global features folders '
                              'examples dataset/global_features/apgem dataset/global_features/delg'))
    parser.add_argument('-lfeat', '--local-features', nargs='+', default=[],
                        help=('paths to the local features folders '
                              'examples dataset/local_features/r2d2 dataset/local_features/d2_tf'))
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    upgrade_1_0_to_1_1_orphan_features(args.local_features, args.global_features)


if __name__ == '__main__':
    upgrade_1_0_to_1_1_orphan_features_command_line()
