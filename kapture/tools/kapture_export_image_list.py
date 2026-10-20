#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Export the list of images of a kapture into a file listing the image files, and optionally their camera parameters.
"""

import logging
import os
# kapture
import kapture
from kapture.utils.cli import run_cli, setup_logging, Verbosity, DEFAULT_VERBOSITY
from kapture.io.csv import kapture_from_dir
from kapture.utils.paths import safe_remove_file


logger = logging.getLogger('export_image_list')


def export_image_list(input: str,
                      output: str,
                      camera_params: bool = False,
                      force: bool = False,
                      verbose: Verbosity = DEFAULT_VERBOSITY,
                      quiet: bool = False) -> None:
    """
    Export kapture data image list to a text file: one image name per line, optionally followed
    by the camera type and parameters (eg. image_name SIMPLE_RADIAL w h f cx cy r).

    :param input: top directory of the kapture
    :param output: path of the image list file
    :param camera_params: if True, add camera parameters after every file name
    :param force: silently overwrite image list file if already exists.
    :param verbose: verbosity level (debug, info, warning, critical, ... or int value)
    :param quiet: silence all logs
    """
    setup_logging(logger, verbose, quiet)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    safe_remove_file(output, force)
    skip_heavy_useless = [kapture.Trajectories,
                          kapture.RecordsLidar, kapture.RecordsWifi,
                          kapture.RecordsAccelerometer, kapture.RecordBluetooth,
                          kapture.RecordGnss, kapture.RecordGyroscope,
                          kapture.RecordsDepth, kapture.RecordsRange, kapture.RecordsMagnetic,
                          kapture.Keypoints, kapture.Descriptors, kapture.GlobalFeatures,
                          kapture.Matches, kapture.Points3d, kapture.Observations]
    kapture_to_export = kapture_from_dir(input, skip_list=skip_heavy_useless)
    output_content = []
    logger.info('starting conversion...')
    for _, sensor_id, filename in kapture.flatten(kapture_to_export.records_camera, is_sorted=True):
        line = filename
        if camera_params:
            camera = kapture_to_export.sensors[sensor_id]
            assert isinstance(camera, kapture.Camera)
            line += ' ' + ' '.join(camera.sensor_params)
        output_content.append(line)

    logger.info('writing exported data...')
    with open(output, 'w') as fid:
        fid.write('\n'.join(output_content))


def main():
    run_cli(export_image_list, aliases={'--camera_params': ['-c']})


if __name__ == '__main__':
    main()
