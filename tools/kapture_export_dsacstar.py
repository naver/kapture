#!/usr/bin/env python3
# Copyright 2022-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os.path
import os.path as path
import path_to_kapture  # enables import kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import get_all_tar_handlers, kapture_from_dir
from kapture.core.Sensors import CameraType
import quaternion
from PIL import Image
import math
import torch
import numpy as np


logger = logging.getLogger('dsacstar')

def get_focal_length(
        sensors: kapture.Sensors,
        sensor_id: str
) -> float:
    assert sensors[sensor_id].sensor_type == 'camera'

    cp = sensors[sensor_id].camera_params
    cam_type = sensors[sensor_id].camera_type
    assert cam_type is not CameraType.UNKNOWN_CAMERA

    if cam_type in [CameraType.FOV, CameraType.OPENCV, CameraType.FULL_OPENCV, CameraType.OPENCV_FISHEYE, CameraType.PINHOLE, CameraType.THIN_PRISM_FISHEYE]:
        return (cp[2] + cp[3]) / 2
    else:
        return cp[2]

def export_dsacstar(
        kapture_root_dir: str,
        dsacstar_root_dir: str
) -> None:
    """
    Export the kapture data to an openSfM format

    :param kapture_root_dir: full path to the top kapture directory
    :param dsacstar_root_dir: path of the directory where to store the data in dsacstar format
    """

    target_height = 480  # rescale images
    nn_subsampling = 8  # sub sampling of our CNN architecture, for size of the initalization targets

    unused = [kapture.RecordsLidar, kapture.RecordsWifi,
              kapture.RecordsAccelerometer, kapture.RecordBluetooth,
              kapture.RecordGnss, kapture.RecordGyroscope,
              kapture.RecordsMagnetic,
              kapture.Descriptors, kapture.GlobalFeatures,
              kapture.Matches]

    with get_all_tar_handlers(kapture_root_dir) as tar_handlers:
        kdata = kapture_from_dir(kapture_root_dir, tar_handlers=tar_handlers, skip_list=unused)
        assert isinstance(kdata, kapture.Kapture)
        assert kdata.records_camera is not None
        assert kdata.trajectories is not None
        assert kdata.sensors is not None

        if kdata.points3d is not None:
            target_output_folder = path.join(dsacstar_root_dir, 'init')
            os.makedirs(target_output_folder, exist_ok=True)

            pts_dict = {}
            for keypoint_type in kdata.keypoints:
                for point3d_idx in kdata.observations:
                    for image_path, keypoint_id in kdata.observations[point3d_idx, keypoint_type]:
                        if image_path not in pts_dict:
                            pts_dict[image_path] = []
                        pt = list(kdata.points3d[point3d_idx][0:3])
                        pt.append(1.0)
                        pts_dict[image_path].append(pt)

        img_output_folder = path.join(dsacstar_root_dir, 'rgb')
        cal_output_folder = path.join(dsacstar_root_dir, 'calibration')
        pose_output_folder = path.join(dsacstar_root_dir, 'poses')

        os.makedirs(img_output_folder, exist_ok=True)
        os.makedirs(cal_output_folder, exist_ok=True)
        os.makedirs(pose_output_folder, exist_ok=True)

        for ts, sensor_id, image_name in kapture.flatten(kdata.records_camera):
            print(f'Exporting {image_name}')

            src_image = path.join(kapture_root_dir, 'sensors/records_data', image_name)
            trg_image = path.join(img_output_folder, image_name.replace('/', '_'))

            im = Image.open(src_image)
            width, height = im.size

            img_aspect = height / width
            if img_aspect > 1:
                # portrait
                img_w = target_height
                img_h = int(math.ceil(target_height * img_aspect))
            else:
                # landscape
                img_w = int(math.ceil(target_height / img_aspect))
                img_h = target_height

            out_w = int(math.ceil(img_w / nn_subsampling))
            out_h = int(math.ceil(img_h / nn_subsampling))

            out_scale = out_w / width
            img_scale = img_w / width

            out_tensor = torch.zeros((3, out_h, out_w))
            out_zbuffer = torch.zeros((out_h, out_w))

            cam_pose = kdata.trajectories[ts][sensor_id]
            cam_pose_inv = cam_pose.inverse()
            R = quaternion.as_rotation_matrix(cam_pose_inv.r)
            T = cam_pose_inv.t

            with open(path.join(pose_output_folder, image_name[:-3].replace('/', '_') + 'txt'), 'w') as f:
                f.write(str(float(R[0, 0])) + ' ' + str(float(R[0, 1])) + ' ' + str(float(R[0, 2])) + ' ' + str(float(T[0, 0])) + '\n')
                f.write(str(float(R[1, 0])) + ' ' + str(float(R[1, 1])) + ' ' + str(float(R[1, 2])) + ' ' + str(float(T[1, 0])) + '\n')
                f.write(str(float(R[2, 0])) + ' ' + str(float(R[2, 1])) + ' ' + str(float(R[2, 2])) + ' ' + str(float(T[2, 0])) + '\n')
                f.write(str(float(0)) + ' ' + str(float(0)) + ' ' + str(float(0)) + ' ' + str(float(1)) + '\n')

            new_image = im.resize((img_w, img_h))
            new_image.save(trg_image)

            cam_pose_np = np.concatenate((quaternion.as_rotation_matrix(cam_pose.r), cam_pose.t), axis=1)
            cam_pose_np = np.concatenate((cam_pose_np, [[0, 0, 0, 1]]), axis=0)
            cam_pose_np = torch.tensor(cam_pose_np).float()

            focal_length = get_focal_length(kdata.sensors, sensor_id)
            with open(path.join(cal_output_folder, image_name[:-3].replace('/', '_') + 'txt'), 'w') as f:
                f.write(str(focal_length * img_scale))

            if kdata.points3d is not None:
                pts_3D = torch.tensor(pts_dict[image_name])
                for pt_idx in range(0, pts_3D.size(0)):
                    scene_pt = pts_3D[pt_idx]
                    scene_pt = scene_pt.unsqueeze(0)
                    scene_pt = scene_pt.transpose(0, 1)

                    # scene to camera coordinates
                    cam_pt = torch.mm(cam_pose_np, scene_pt)
                    # projection to image
                    img_pt = cam_pt[0:2, 0] * focal_length / cam_pt[2, 0] * out_scale

                    y = img_pt[1] + out_h / 2
                    x = img_pt[0] + out_w / 2

                    x = int(torch.clamp(x, min=0, max=out_tensor.size(2) - 1))
                    y = int(torch.clamp(y, min=0, max=out_tensor.size(1) - 1))

                    if cam_pt[2, 0] > 1000:  # filter some outlier points (large depth)
                        continue

                    if out_zbuffer[y, x] == 0 or out_zbuffer[y, x] > cam_pt[2, 0]:
                        out_zbuffer[y, x] = cam_pt[2, 0]
                        out_tensor[:, y, x] = pts_3D[pt_idx, 0:3]

                torch.save(out_tensor, path.join(target_output_folder, image_name[:-3].replace('/', '_') + 'dat'))


def export_dsacstar_command_line():
    """
    Exports dsacstar format from kapture.
    """
    parser = argparse.ArgumentParser(description='Exports from kapture to dsacstar')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    ####################################################################################################################
    parser.add_argument('-k', '--kapture', required=True,
                        help='path to kapture root directory')
    parser.add_argument('-o', '--dsacstar', required=True,
                        help='directory where to save dsacstar files')
    # parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
    #                     help='Force delete dsacstar data if already exists.')
    args = parser.parse_args()
    ####################################################################################################################
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # for debug, let kapture express itself.
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug('\\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()
        if k != 'command'))

    args.kapture = path.normpath(path.abspath(args.kapture))
    args.dsacstar = path.normpath(path.abspath(args.dsacstar))
    export_dsacstar(args.kapture, args.dsacstar)

if __name__ == '__main__':
    export_dsacstar_command_line()
