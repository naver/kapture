# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Represents a 6 dimensions pose.
"""

import math
import numpy as np
import quaternion
from typing import List, Optional
from numba import njit


class PoseTransform:
    """
    Brief: Transforms points from reference frame to another.
            In case of
             - Trajectories: transforms points from world to device,
             - Rig: transforms points from rig to camera.
    """

    def __init__(self, r=[1, 0, 0, 0], t=[0, 0, 0]):
        """
        Creates a Pose transform (eg. world to device) with given rotation + translation

        :param r: rotation part expressed as a quaternion (a list of floats, a numpy array, or a numpy quaternion)
        :param t: translation part expressed as a vector ((a list of floats, a numpy array)
        """

        # quaternion
        if isinstance(r, quaternion.quaternion):
            # already a quaternion : copy aziz (lumière)
            self._r = r
        elif isinstance(r, (list, np.ndarray, np.generic)):
            # its a list or numpy => convert to quaternion if valid
            self._r = quaternion.from_float_array(r)
        else:  # unknown entry
            self._r = None

        if self._r is not None and np.isnan(np.sum(self._r)):
            # check no nan in the array (meaning failure)
            # sum trick from https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
            self._r = None

        # translation
        if isinstance(t, (np.ndarray, np.generic)):
            #  numpy : copy aziz lumière
            self._t = t
        elif isinstance(t, list):
            # its a list: convert to numpy if valid
            self._t = np.array(t, dtype=float)
        else:
            self._t = None

        if self._t is not None:
            assert isinstance(self._t, (np.ndarray, np.generic))
            if np.isnan(np.sum(self._t)):
                # check no nan in the array (meaning failure)
                # sum trick from https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
                self._t = None
            else:
                # make sure af the shape of t
                self._t = self._t.reshape((3, 1))

    @property
    def r(self):
        """
        :return: rotation as quaternion
        """
        return self._r

    @property
    def r_raw(self) -> Optional[List[float]]:
        """
        :return: rotation as list of float
        """
        return quaternion.as_float_array(self._r).tolist() if self._r is not None else None

    @property
    def t(self):
        """
        :return: translation as numpy ndarray
        """
        return self._t

    @property
    def t_raw(self) -> Optional[List[float]]:
        """
        :return: translation as list of float
        """
        return self._t.flatten().tolist() if self._t is not None else None

    def inverse(self) -> 'PoseTransform':
        """
        :return: inverted pose
        """
        assert self._r is not None
        assert self._t is not None
        r_inv = self.r.inverse()
        rotation_inv_matrix = np.empty((3, 3), dtype=float)
        rotation_inv_as_np = np.array([r_inv.w, r_inv.x, r_inv.y, r_inv.z])
        _as_rotation_matrix_njit(rotation_inv_as_np, rotation_inv_matrix)
        t_inv = np.matmul(rotation_inv_matrix, self.t * -1.0)
        pose_inv = PoseTransform.__new__(PoseTransform)
        pose_inv._r = r_inv
        pose_inv._t = t_inv
        return pose_inv

    def rescale(self, scale: float):
        """
        rescales translation part
        :param scale:
        :return: scaled pose
        """
        if self._t is not None:
            self._t = self._t * scale

    @staticmethod
    def compose(pose_list: List['PoseTransform']) -> 'PoseTransform':
        """
        Merges multiple pose transformation into a single one.

        :param pose_list: the list of pose transformation to be merged. They are merged from right to left.
        :return: the pose transformation being the composition of the given ones.
        """
        assert isinstance(pose_list, list)
        pose_composed = pose_list[0]
        for pose_current in pose_list[1:]:
            # shrink the poses from the left
            # assert pose_current._r is not None
            # assert pose_current._t is not None
            new_r = pose_composed.r * pose_current.r
            rotation_matrix = np.empty((3, 3), dtype=float)
            rotation_composed_as_np = np.array([pose_composed.r.w, pose_composed.r.x,
                                                pose_composed.r.y, pose_composed.r.z])
            _as_rotation_matrix_njit(rotation_composed_as_np, rotation_matrix)
            new_t = np.add(np.matmul(rotation_matrix, pose_current.t), pose_composed.t)

            pose_composed = PoseTransform.__new__(PoseTransform)
            pose_composed._r = new_r
            pose_composed._t = new_t
        return pose_composed

    def transform_points(self, points3d: np.ndarray) -> np.ndarray:
        """
        Applies the self transformation to the given 3d points.

        :param points3d: input 3d points, stored row wise (one point per row)
        :return: another array with the 3d points transformed using the PoseTransform.
        """
        assert self._r is not None
        assert self._t is not None
        assert isinstance(points3d, np.ndarray)

        if points3d.shape[1] == 6:  # expunge RGB
            points3d = points3d[:, 0:3]
        points3d = points3d.transpose()
        rotation_matrix = np.empty((3, 3), dtype=float)
        rotation_as_np = np.array([self.r.w, self.r.x, self.r.y, self.r.z])
        _as_rotation_matrix_njit(rotation_as_np, rotation_matrix)
        points3d = np.add(np.matmul(rotation_matrix, points3d), self.t)
        return points3d.transpose()

    def __repr__(self) -> str:
        return 'r:{},  t:{}'.format(self.r_raw, self.t_raw)

    def __eq__(self, other) -> bool:
        if (self._t is None and other.t is not None) or (self._t is not None and other.t is None):
            return False
        if (self._r is None and other.r is not None) or (self._r is not None and other.r is None):
            return False
        self_trans = self.t_raw
        other_trans = other.t_raw
        for left, right in zip(self_trans, other_trans):
            # Check distances at 10-2 millimeters
            if not math.isclose(left, right, rel_tol=1.e-05, abs_tol=1.e-05):
                return False
        self_rot = self.r_raw
        other_rot = other.r_raw
        for left, right in zip(self_rot, other_rot):
            # Check angles at 10-2 degrees
            if not math.isclose(left, right, rel_tol=1.e-02, abs_tol=1.e-02):
                return False
        return True


# https://github.com/moble/quaternion/blob/main/src/quaternion/__init__.py#L200
@njit
def _as_rotation_matrix_njit(q, rot_mat):
    q_norm = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    if abs(q_norm-1.0) < 1e-14:
        rot_mat[0, 0] = 1 - 2*(q[2]**2 + q[3]**2)
        rot_mat[0, 1] = 2*(q[1]*q[2] - q[3]*q[0])
        rot_mat[0, 2] = 2*(q[1]*q[3] + q[2]*q[0])
        rot_mat[1, 0] = 2*(q[1]*q[2] + q[3]*q[0])
        rot_mat[1, 1] = 1 - 2*(q[1]**2 + q[3]**2)
        rot_mat[1, 2] = 2*(q[2]*q[3] - q[1]*q[0])
        rot_mat[2, 0] = 2*(q[1]*q[3] - q[2]*q[0])
        rot_mat[2, 1] = 2*(q[2]*q[3] + q[1]*q[0])
        rot_mat[2, 2] = 1 - 2*(q[1]**2 + q[2]**2)
    else:
        rot_mat[0, 0] = 1 - 2*(q[2]**2 + q[3]**2)/q_norm
        rot_mat[0, 1] = 2*(q[1]*q[2] - q[3]*q[0])/q_norm
        rot_mat[0, 2] = 2*(q[1]*q[3] + q[2]*q[0])/q_norm
        rot_mat[1, 0] = 2*(q[1]*q[2] + q[3]*q[0])/q_norm
        rot_mat[1, 1] = 1 - 2*(q[1]**2 + q[3]**2)/q_norm
        rot_mat[1, 2] = 2*(q[2]*q[3] - q[1]*q[0])/q_norm
        rot_mat[2, 0] = 2*(q[1]*q[3] - q[2]*q[0])/q_norm
        rot_mat[2, 1] = 2*(q[2]*q[3] + q[1]*q[0])/q_norm
        rot_mat[2, 2] = 1 - 2*(q[1]**2 + q[2]**2)/q_norm
    return
