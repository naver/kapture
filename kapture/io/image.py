# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import os
import os.path as path
from random import randint
import numpy as np
from PIL import Image, ImageDraw

from .features import image_keypoints_from_file


def image_keypoints_to_image(
        image: Image,
        keypoints: np.array,
        radius: int = 2,
        filled: bool = True) -> Image:
    """
    Displays keypoints on top of the image.

    :param image: an image
    :param keypoints: the keypoints
    :param radius: radius of the drawn circles
    :param filled: True: draw discs, False: draw circles
    :return: a new Image
    """
    draw = ImageDraw.Draw(image)
    for coords in keypoints[:, 0:2]:
        coords_tl = (coords - radius).astype(int)
        coords_br = (coords + radius).astype(int)
        color = tuple([randint(0, 255) for _ in range(3)])
        draw.ellipse((coords_tl[0], coords_tl[1], coords_br[0], coords_br[1]), outline=color,
                     fill=color if filled else None)
    return image


def image_keypoints_to_image_file(
        output_filepath: str,
        image_filepath: str,
        keypoints_filepath: str,
        keypoint_dtype,
        keypoint_dsize,
        radius: int = 2) -> None:
    """
    Displays keypoints on top of the image and save it to image file.

    :param output_filepath: input path to output image of keypoints.
    :param image_filepath: input path to input image.
    :param keypoints_filepath: input path to keypoints file.
    :param keypoint_dtype: input data type of keypoints data (cf. binary).
    :param keypoint_dsize: input data size of keypoints data (cf. binary)
    :param radius: radius of the keypoint in image (in pixel).
    :return:
    """
    os.makedirs(path.dirname(output_filepath), exist_ok=True)
    keypoints = image_keypoints_from_file(keypoints_filepath, keypoint_dtype, keypoint_dsize)
    image = Image.open(image_filepath)
    image_of_keypoints = image_keypoints_to_image(image, keypoints, radius)
    image_of_keypoints.save(output_filepath)
