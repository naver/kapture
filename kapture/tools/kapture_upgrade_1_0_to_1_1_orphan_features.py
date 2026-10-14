#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data in version 1.0 to version 1.1 (only orphan features)
"""

import logging
from typing import List

from kapture.utils.upgrade import upgrade_1_0_to_1_1_orphan_features
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('upgrade_1_0_to_1_1_orphan_features')


def upgrade_orphan_features(local_features: List[str] = [],
                            global_features: List[str] = [],
                            verbose: Verbosity = DEFAULT_VERBOSITY,
                            quiet: bool = False) -> None:
    """
    Upgrade orphan features to kapture 1.1. Orphan features are features stored outside the kapture folder,
    they must follow the kapture-localization recommendation
    https://github.com/naver/kapture-localization/blob/main/doc/tutorial.adoc#recommended-dataset-structure

    :param local_features: paths to the local features folders,
                           examples dataset/local_features/r2d2 dataset/local_features/d2_tf
    :param global_features: paths to the global features folders,
                            examples dataset/global_features/apgem dataset/global_features/delg
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    upgrade_1_0_to_1_1_orphan_features(local_features, global_features)


def main():
    run_cli(upgrade_orphan_features, aliases={'--local_features': ['-lfeat'], '--global_features': ['-gfeat']})


if __name__ == '__main__':
    main()
