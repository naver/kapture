# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import os.path as path
from random import randint
from typing import Dict
import numpy as np

import kapture
from .features import image_keypoints_from_file

PLY_HEADER_TEMPLATE = '\n'.join([
    'ply',
    'format ascii 1.0',
    'element vertex {nb_vertex}',
    'property double x',
    'property double y',
    'property double z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'element edge {nb_edges}',
    'property int vertex1',
    'property int vertex2',
    'end_header'])

########################################################################################################################
BLACK = 3 * [0]
WHITE = 3 * [255]
GREY = 3 * [127]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
AXIS_COLORS = [GREY, RED, GREEN, BLUE]


########################################################################################################################
def get_axis_in_world(
        pose_device_from_world: kapture.PoseTransform, length: float = 1.0) -> kapture.Points3d:
    """
    Returns a quadruplet of points (0,x,y,z) representing the axis of the device into the world.

    :param pose_device_from_world: assume the transformation is device from world.
    :param length: distance between each axis point and center.
    :return: 4 points (center, x, y, z) arranged by cols
    """
    assert (isinstance(pose_device_from_world, kapture.PoseTransform))
    sensor_axis = np.array([
        [0, 0, 0],  # 0
        [length, 0, 0],  # X
        [0, length, 0],  # Y
        [0, 0, length],  # Z
    ])
    return pose_device_from_world.inverse().transform_points(sensor_axis)


def header_to_ply_stream(stream, nb_vertex: int = 0, nb_edges: int = 0) -> None:
    """
    Writes PLY header to a stream.

    :param stream: an open stream to write to
    :param nb_vertex: number of vertex
    :param nb_edges: number of edges
    """
    stream.write(PLY_HEADER_TEMPLATE.format(nb_vertex=nb_vertex,
                                            nb_edges=nb_edges))
    stream.write('\n')


def rig_to_ply_stream(stream, rig: Dict[str, kapture.PoseTransform], axis_length: float = 1.) -> None:
    """
    Writes the rig to a stream.

    :param stream: an open stream to write to
    :param rig: rig to write
    :param axis_length: length of the axis
    """
    device_list = [(cam_id, pose_tr)
                   for cam_id, pose_tr in rig.items()]

    # add the rig center in devices
    device_list = [(-1, kapture.PoseTransform())] + device_list

    # create 4 points per device: 1 for center, 3 for axis
    points_colored_list = []
    edges_list = []
    for cam_id, pose_tr in device_list:
        axis = get_axis_in_world(pose_tr, axis_length)
        edges_list += [(0, len(points_colored_list))]
        points_colored_list += [p + AXIS_COLORS[i] for i, p in enumerate(axis.tolist())]

    # write points into ply
    header_to_ply_stream(stream,
                         nb_vertex=len(points_colored_list),
                         nb_edges=len(edges_list))
    for p3d in points_colored_list:
        line = ['{:<25}'.format(i) for i in p3d[0:3]]
        line += ['{:03}'.format(int(i)) for i in p3d[3:6]]
        stream.write(' '.join(line) + '\n')
    for e in edges_list:
        line = ['{:2}'.format(i) for i in e]
        stream.write(' '.join(line) + '\n')


def rig_to_ply(filepath: str, rig: Dict[str, kapture.PoseTransform], axis_length: float = 1.) -> None:
    """
    Writes the rig to a file.

    :param filepath: file path to write to
    :param rig: rig to write
    :param axis_length: length of the axis
    """
    with open(filepath, 'w') as f:
        rig_to_ply_stream(f, rig, axis_length)


########################################################################################################################
def trajectories_to_ply_stream(stream, trajectories: kapture.Trajectories, axis_length: float = 1.) -> None:
    """
    Writes the trajectories to a stream.
     trajectories[ts][device_id] = [pose]

    :param stream: an open stream to write to
    :param trajectories: trajectories to write
    :param axis_length: length of the axis
    """
    pose_list = (pose_tr
                 for _, _, pose_tr in kapture.flatten(trajectories, is_sorted=True)
                 if not np.any(np.isnan(pose_tr.t)))  # filter out if no position

    # create 4 points per pose: 1 for center, 3 for axis
    points_colored_list = []
    for pose_tr in pose_list:
        axis = get_axis_in_world(pose_tr, axis_length)
        points_colored_list += [p + AXIS_COLORS[i] for i, p in enumerate(axis.tolist())]

    # write points into ply
    header_to_ply_stream(stream, nb_vertex=len(points_colored_list))
    for p3d in points_colored_list:
        line = ['{:<25}'.format(i) for i in p3d[0:3]]
        line += ['{:<4}'.format(i) for i in p3d[3:6]]
        stream.write(' '.join(line) + '\n')


def trajectories_to_ply(
        filepath: str,
        trajectories: kapture.Trajectories,
        axis_length: float = 1.
):
    """
    Writes trajectory to PLY format (for visualization).
    Each pose in trajectory leads to a ply dot. 3 additional points are added in X (red), Y (Green) and Z (blue)
    direction around each pose.

    :param filepath: input ply file path.
    :param trajectories: input trajectory
    :param axis_length: length of axis representing the orientation of each pose in trajectory.
    :return:
    """
    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as fout:
        trajectories_to_ply_stream(fout, trajectories, axis_length)


def points3d_to_stream(stream, points3d: kapture.Points3d) -> None:
    """
    Writes the 3D points to a stream.

    :param stream: an open stream to write to
    :param points3d: 3d points to write
    """
    header_to_ply_stream(stream, nb_vertex=len(points3d))
    for p3d in points3d:
        line = ['{:20}'.format(i) for i in p3d[0:3]] + ['{:03d}'.format(int(i)) for i in p3d[3:6]]
        stream.write('  '.join(line) + '\n')


def points3d_to_ply(filepath: str, points3d: kapture.Points3d) -> None:
    """
    Writes 3D points into ply file.

    :param filepath: ply file path.
    :param points3d: 3D points.
    """
    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        points3d_to_stream(f, points3d)


def image_keypoints_to_stream(stream, image_keypoints: np.array) -> None:
    """
    Plots image keypoints onto a 2D plane. Use random colors.

    :param stream: an open stream to write to
    :param image_keypoints: the image keypoints to write
    """
    header_to_ply_stream(stream, nb_vertex=len(image_keypoints))
    for kpt in image_keypoints:
        coords = kpt[0:2]
        line = ['{:20}'.format(i) for i in coords] + ['0.0'] + ['{:03d}'.format(randint(0, 255)) for _ in range(3)]
        stream.write('  '.join(line) + '\n')


def image_keypoints_to_ply(ply_filepath: str, image_keypoints_filepath: str, keypoint_dtype, keypoint_dsize) -> None:
    """
    Plots image keypoints onto a 2D plane.

    :param ply_filepath: path to the ply file to write
    :param image_keypoints_filepath: path to the image keypoints file to read
    :param keypoint_dtype: keypoint data type
    :param keypoint_dsize: keypoint data size
    """
    os.makedirs(path.dirname(ply_filepath), exist_ok=True)
    image_keypoints = image_keypoints_from_file(image_keypoints_filepath, keypoint_dtype, keypoint_dsize)
    with open(ply_filepath, 'w') as f:
        image_keypoints_to_stream(f, image_keypoints)
