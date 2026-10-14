#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Convert kapture data in version 1.0 to version 1.1 inplace
"""

import logging
from typing import Optional
# import numpy as np like in kapture.io.csv
# so that types written as "np.float32" are understood by read_old_image_features_csv
import numpy as np  # noqa: F401

from kapture.utils.upgrade import upgrade_1_0_to_1_1_inplace
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY

logger = logging.getLogger('upgrade_1_0_to_1_1_inplace')


def upgrade_inplace(input: str,
                    keypoints_type: Optional[str] = None,
                    descriptors_type: Optional[str] = None,
                    global_features_type: Optional[str] = None,
                    descriptors_metric_type: str = 'L2',
                    global_features_metric_type: str = 'L2',
                    verbose: Verbosity = DEFAULT_VERBOSITY,
                    quiet: bool = False) -> None:
    """
    Convert kapture data in version 1.0 to version 1.1 inplace.

    :param input: path to kapture 1.0 directory, upgraded in place
    :param keypoints_type: type of keypoints (read from input if not given)
    :param descriptors_type: type of descriptors (read from input if not given)
    :param global_features_type: type of global features (read from input if not given)
    :param descriptors_metric_type: metric type of descriptors
    :param global_features_metric_type: metric type of global features
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    upgrade_1_0_to_1_1_inplace(input,
                               keypoints_type, descriptors_type, global_features_type,
                               descriptors_metric_type, global_features_metric_type)


def main():
    run_cli(upgrade_inplace)


if __name__ == '__main__':
    main()
