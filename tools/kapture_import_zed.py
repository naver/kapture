#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
import zed (stereolabs) camera recording from svo file and imu.csv.
"""

import argparse
import logging
import os
import os.path as path
import re
import numpy as np
import quaternion
from typing import Optional
from PIL import Image
from tqdm import tqdm
# kapture
import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
import kapture.io.features
from kapture.io.records import TransferAction, get_image_fullpath
from kapture.utils.paths import path_secure

logger = logging.getLogger('kapture_import_zed')


try:
    import pyzed.sl as sl
except ImportError as e:
    logger.critical('***** pyzed API MUST be installed ******\n'
                    'see: https://www.stereolabs.com/docs/installation/')
    raise


class ImportError(Exception):
    pass


def kapture_import_zed(
        destination_kapture_dir_path: str,
        svo_file_path: str,
        sensor_file_path: Optional[str] = None,
        force_overwrite_existing: bool = False
) -> None:
    """
    Imports data from silda dataset.

    :param destination_kapture_dir_path: path to the output kapture directory
    :param svo_file_path: input path to svo file
    :param sensor_file_path: input path to imu.csv file
    :param force_overwrite_existing: if true, Silently overwrite kapture files if already exists.
    :param images_import_strategy: how to copy image files.
    """
    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE  # same as kapture
    init_params.coordinate_units = sl.UNIT.METER  # Use milliliter units (for depth measurements)

    camera_names = ['Camera1', 'Camera2']
    rig_name = 'zed_cam'

    # Create ZED objects
    zed = sl.Camera()
    kapture_sensors = kapture.Sensors()
    kapture_records_camera = kapture.RecordsCamera()
    kapture_trajectories = kapture.Trajectories()
    kapture_rigs = kapture.Rigs()
    kapture_accelerometer = kapture.RecordsAccelerometer() if sensor_file_path else None
    kapture_gyroscope = kapture.RecordsGyroscope() if sensor_file_path else None
    try:
        # Open the SVO file specified as a parameter
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise ImportError('unable to open cam')

        # calib
        configuration_sensors = zed.get_camera_information().sensors_configuration
        configuration_cameras = zed.get_camera_information().camera_configuration
        camera_intrinsics = [configuration_cameras.calibration_parameters.left_cam,
                             configuration_cameras.calibration_parameters.right_cam]
        camera_extrinsics = configuration_cameras.calibration_parameters.stereo_transform
        nb_frames = zed.get_svo_number_of_frames()
        image_size = zed.get_camera_information().camera_resolution
        width = image_size.width
        height = image_size.height

        logger.info('write camera intrinsic sensors')
        for intrinsics, cam_id in zip(camera_intrinsics, camera_names):
            camera_params = [width, height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
            kapture_sensors[cam_id] = kapture.Camera(camera_type=kapture.CameraType.PINHOLE,
                                                     camera_params=camera_params,
                                                     name=cam_id)
        if sensor_file_path:
            # add the sensors to sensor
            accelerometer_params = [configuration_sensors.accelerometer_parameters.noise_density,
                                    configuration_sensors.accelerometer_parameters.random_walk,
                                    configuration_sensors.accelerometer_parameters.resolution]
            kapture_sensors['Accelerometer'] = kapture.Sensor(sensor_type='accelerometer', name='Accelerometer',
                                                              sensor_params=accelerometer_params)
            gyroscope_params = [configuration_sensors.gyroscope_parameters.noise_density,
                                configuration_sensors.gyroscope_parameters.random_walk,
                                configuration_sensors.gyroscope_parameters.resolution]
            kapture_sensors['Gyroscope'] = kapture.Sensor(sensor_type='gyroscope', name='Gyroscope',
                                                          sensor_params=gyroscope_params)

        logger.info('write extrinsic to rigs')
        # rigs[rig_id, sensor_id] = <PoseTransform>
        kapture_rigs[rig_name, camera_names[0]] = kapture.PoseTransform()
        kapture_rigs[rig_name, camera_names[1]] = kapture.PoseTransform(t=configuration_cameras.calibration_parameters.T)

        if sensor_file_path:
            # add the imu to rig config
            logger.info('write imu')
            imu_from_cam_mat = configuration_sensors.camera_imu_transform.m
            imu_from_cam_rot = quaternion.from_rotation_matrix(imu_from_cam_mat[0:3, 0:3])
            imu_from_cam_trans = imu_from_cam_mat[0:3, 3]
            imu_from_cam_pose = kapture.PoseTransform(r=imu_from_cam_rot, t=imu_from_cam_trans)
            # the imu_from_cam_pose is incomplete, because not expressed in sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            kapture_rigs[rig_name, 'Accelerometer'] = imu_from_cam_pose
            kapture_rigs[rig_name, 'Gyroscope'] = imu_from_cam_pose
        tracking_parameters = sl.PositionalTrackingParameters(_init_pos=sl.Transform())
        status = zed.enable_positional_tracking(tracking_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise ImportError('unable to track')

        # Prepare side by side image container equivalent to CV_8UC4
        # Prepare single image containers
        image = sl.Mat()
        zed_pose = sl.Pose()

        for cam_id in camera_names:
            image_dir = path.dirname(get_image_fullpath(kapture_dir_path=destination_kapture_dir_path,
                                                        image_filename=f'{cam_id}/0.png'))
            os.makedirs(image_dir, exist_ok=True)

        logger.info('write images and trajectories')
        runtime_parameters = sl.RuntimeParameters()

        imu_file = open(sensor_file_path, 'rt') if sensor_file_path else None
        imu_lines = (l.split(',') for l in imu_file.readlines() if not l.startswith('#'))
        for i in tqdm(range(nb_frames)):
            status = zed.grab(runtime_parameters)
            if status != sl.ERROR_CODE.SUCCESS:
                raise ImportError()

            svo_frame_idx = zed.get_svo_position()
            assert svo_frame_idx == i
            # Retrieve SVO images
            status = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) # Get the pose of the left eye of the camera with reference to the world frame
            # ignore status, its faulty
            timestamp_ns = zed_pose.timestamp.get_nanoseconds()

            for cam_id, side in zip(camera_names, [sl.VIEW.LEFT, sl.VIEW.RIGHT]):
                status = zed.retrieve_image(image, side)
                if status != sl.ERROR_CODE.SUCCESS:
                    raise ImportError(f'unable to retrieve_image {cam_id}')

                image_name = path.join(cam_id, f'{timestamp_ns}.png')
                # records_camera[timestamp, sensor_id] = image_name
                kapture_records_camera[timestamp_ns, cam_id] = image_name

                image_file_path = get_image_fullpath(kapture_dir_path=destination_kapture_dir_path,
                                                     image_filename=image_name)
                image.write(image_file_path)

            # if status != sl.ERROR_CODE.SUCCESS:
            #     raise ExportError('unable to get_position')
            # --> linear_acceleration returns correct values (norm = 9.8)
            # --> angular_velocity is at 0 as expected because the SVO contains only data that have a meaning in image framerate (i.e. accelerometer + magnetometer)
            # print(imu.get_angular_velocity(), imu.get_linear_acceleration())
            pose_cam_from_world = zed_pose.pose_data(sl.Transform())
            pose_cam_from_world.inverse()
            orientation = pose_cam_from_world.get_orientation().get()  # x, y, z, w
            translation = pose_cam_from_world.get_translation().get()  # x, y, z
            # trajectories[(timestamp, sensor_id)] = < PoseTransform >
            orientation = orientation[[3, 0, 1, 2]]  # reorder w, x, y, z
            kapture_trajectories[timestamp_ns, rig_name] = kapture.PoseTransform(r=orientation, t=translation)

            # grady on imu till we reach current timestamp
            if imu_lines:
                """
                full sensor recording is :
                    idx,imu_Timestamp[sec],mag_Timestamp[sec],baro_Timestamp[sec],
                    accX[m/s^2],accY[m/s^2],accZ[m/s^2],
                    gyroX[deg/s],gyroY[deg/s],gyroZ[deg/s],
                    magX[uT],magY[uT],magZ[uT],
                    orX[deg],orY[deg],orZ[deg],
                    press[hPa],rel_alt[m],moving,
                    temp_left[C],temp_right[C],temp_imu[C],temp_barom[C],
                see : https://www.stereolabs.com/docs/gstreamer/zed-data-csv-sink/
                """
                for sensor_record in imu_lines:
                    imu_timestamp_ns = int(float(sensor_record[1]) * 1e9)
                    accelerometer_ms2 = [float(v) for v in sensor_record[4:7]]
                    gyroscope_rads = [np.deg2rad(float(v)) for v in sensor_record[7:10]]
                    kapture_accelerometer[imu_timestamp_ns, 'Accelerometer'] = kapture.RecordAccelerometer(*accelerometer_ms2)
                    kapture_gyroscope[imu_timestamp_ns, 'Gyroscope'] = kapture.RecordGyroscope(*gyroscope_rads)
                    # TODO: read magneto, baro and co
                    # read until the end or until timestamp is ahead of images
                    if imu_timestamp_ns >= timestamp_ns:
                        break

    except ImportError as e:
        logger.critical(e)
    except KeyboardInterrupt:
        logger.critical('interrupted by user')
    except Exception as e:
        logger.critical(e)
    finally:
        zed.disable_positional_tracking()
        zed.close()

        kapture_data = kapture.Kapture(
            sensors=kapture_sensors,
            rigs=kapture_rigs,
            records_camera=kapture_records_camera,
            trajectories=kapture_trajectories,
            records_accelerometer=kapture_accelerometer,
            records_gyroscope=kapture_gyroscope
        )
    logger.info(f'imported kapture: {repr(kapture_data)}')
    kapture_to_dir(kapture_dirpath=destination_kapture_dir_path, kapture_data=kapture_data)


def kapture_import_zed_command_line() -> None:
    """
    Imports zed recording from stereolabs (svo file and or imu.csv).
    """
    parser = argparse.ArgumentParser(
        description='Imports zed recording from stereolabs (svo file and or imu.csv).')
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
    parser.add_argument('-s', '--svo',
                        help='input path svo file')
    parser.add_argument('-u', '--imu',
                        help='input imu file')
    parser.add_argument('-o', '--output', required=True, help='output directory.')
    ####################################################################################################################
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    kapture_import_zed(
        destination_kapture_dir_path=args.output,
        svo_file_path=args.svo,
        sensor_file_path=args.imu,
        force_overwrite_existing=args.force,
    )


if __name__ == '__main__':
    kapture_import_zed_command_line()
