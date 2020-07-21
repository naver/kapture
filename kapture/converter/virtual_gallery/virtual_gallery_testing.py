# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Virtual gallery testing data import.
"""

from collections import OrderedDict
import logging
import quaternion
from typing import Iterable, List
# kapture
import kapture
# local
from .virtual_gallery_constants import virtual_gallery_camera_model, virtual_gallery_width, virtual_gallery_height

logger = logging.getLogger('virtual_gallery')


class VirtualGalleryTestingIntrinsic:
    """
    virtual gallery testing intrinsics
    """

    def __init__(self, light_id: int, occlusion_id: int, frame_id: int, intrinsics: list):
        self.light_id = light_id
        self.occlusion_id = occlusion_id
        self.frame_id = frame_id
        self.intrinsics = intrinsics

    def __hash__(self):
        return hash((self.light_id, self.occlusion_id,  self.frame_id, tuple(self.intrinsics)))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.light_id == other.light_id and
                self.occlusion_id == other.occlusion_id and
                self.frame_id == other.camera_id and
                self.frame_id == other.intrinsics)


class VirtualGalleryTestingExtrinsic:
    """
    virtual gallery testing extrinsics
    """

    def __init__(self, frame_id: int, light_id: int, occlusion_id: int, extrinsics: list):
        self.light_id = light_id
        self.extrinsics = extrinsics
        self.frame_id = frame_id
        self.occlusion_id = occlusion_id

    def __hash__(self):
        return hash((self.light_id, self.occlusion_id, tuple(self.extrinsics), self.frame_id))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.light_id == other.light_id and
                self.occlusion_id == other.occlusion_id and
                self.extrinsics == other.extrinsics and
                self.frame_id == other.frame_id)


def import_testing_intrinsics(input_root: str,
                              light_range: list,
                              occlusion_range: list
                              ) -> List[VirtualGalleryTestingIntrinsic]:
    """
    Import testing intrinsics
    # format of intrinsic.txt is
    # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
    # 0 0 1371.022 1371.022 959.5 539.5

    :param input_root:
    :param light_range:
    :param occlusion_range:
    :return: intrinsics
    """
    # assert that all intrinsics are identical for one camera and take any

    intrinsics = []
    for light_id in light_range:
        for occlusion_id in occlusion_range:
            with open(f'{input_root}/testing/gallery_light{light_id}'
                      f'_occlusion{occlusion_id}/intrinsic.txt', 'r') as file:
                lines = file.readlines()
                for split in map(lambda x: x.split(), lines):
                    if split[0].isdigit():
                        intrinsic = VirtualGalleryTestingIntrinsic(
                            light_id, occlusion_id, int(split[0]),
                            [float(split[2]), float(split[3]), float(split[4]), float(split[5])]
                        )
                        intrinsics.append(intrinsic)
    return intrinsics


def import_testing_extrinsics(input_root: str,
                              light_range: list,
                              occlusion_range: list
                              ) -> List[VirtualGalleryTestingExtrinsic]:
    """
    Import testing extrinsics.

    # format of extrinsic.txt is
    # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0
    # 0 1
    # 0 0 0.929348 0 0.3692049 2.798107 0 1 0 1.65 -0.3692049 0 0.929348
    # 1.618328 0 0 0 1

    :param input_root:
    :param light_range:
    :param occlusion_range:
    :return: extrinsics
    """
    extrinsics = []
    for light_id in light_range:
        for occlusion_id in occlusion_range:
            with open(f'{input_root}/testing/gallery_light{light_id}'
                      f'_occlusion{occlusion_id}/extrinsic.txt', 'r') as file:
                lines = file.readlines()
                for split in map(lambda x: x.split(), lines):
                    if split[0].isdigit():
                        extrinsic = VirtualGalleryTestingExtrinsic(
                            int(split[0]), light_id, occlusion_id,
                            [float(value) for value in split[2:]]
                        )
                        extrinsics.append(extrinsic)
    return extrinsics


def _get_testing_camera_name(light_id: int, occlusion_id: int, frame_id: int) -> str:
    return f'testing_light_{light_id}_occlusion_{occlusion_id}_frame_{frame_id}'


def convert_testing_intrinsics(testing_intrinsics: Iterable[VirtualGalleryTestingIntrinsic],
                               sensors: kapture.Sensors) -> None:
    """
    Import all testing intrinsics into the sensors definitions.

    :param testing_intrinsics: testing intrinsics to import
    :param sensors: list of sensor definitions where to add the new definitions
    """
    logger.info("Converting testing cameras...")
    for intrinsic in testing_intrinsics:
        camera_device_id = _get_testing_camera_name(intrinsic.light_id, intrinsic.occlusion_id, intrinsic.frame_id)
        camera = kapture.Camera(virtual_gallery_camera_model,
                                [virtual_gallery_width, virtual_gallery_height] + intrinsic.intrinsics)
        sensors[camera_device_id] = camera


def convert_testing_extrinsics(offset: int,
                               testing_extrinsics: Iterable[VirtualGalleryTestingExtrinsic],
                               images: kapture.RecordsCamera,
                               trajectories: kapture.Trajectories) -> None:
    """
    Import all testing extrinsics into the images and trajectories.

    :param offset:
    :param testing_extrinsics: testing extrinsics to import
    :param images: image list to add to
    :param trajectories: trajectories to add to
    """
    # Map (light_id, loop_id, frame_id) to a unique timestamp
    testing_frames_tuples = ((extrinsic.light_id, extrinsic.occlusion_id, extrinsic.frame_id)
                             for extrinsic in testing_extrinsics)
    testing_frames_tuples = OrderedDict.fromkeys(testing_frames_tuples).keys()
    testing_frame_mapping = {v: n + offset for n, v in enumerate(testing_frames_tuples)}

    # Export images and trajectories
    logger.info("Converting testing images and trajectories...")
    for extrinsic in testing_extrinsics:
        rotation_matrix = [extrinsic.extrinsics[0:3], extrinsic.extrinsics[4:7], extrinsic.extrinsics[8:11]]
        rotation = quaternion.from_rotation_matrix(rotation_matrix)
        timestamp = testing_frame_mapping[(extrinsic.light_id, extrinsic.occlusion_id, extrinsic.frame_id)]
        camera_device_id = _get_testing_camera_name(extrinsic.light_id, extrinsic.occlusion_id, extrinsic.frame_id)
        translation_vector = [extrinsic.extrinsics[3], extrinsic.extrinsics[7], extrinsic.extrinsics[11]]
        images[(timestamp, camera_device_id)] = (f'testing/gallery_light{extrinsic.light_id}_occlusion'
                                                 f'{extrinsic.occlusion_id}/frames/rgb/'
                                                 f'camera_0/rgb_{extrinsic.frame_id:05}.jpg')
        trajectories[(timestamp, camera_device_id)] = kapture.PoseTransform(rotation, translation_vector)
