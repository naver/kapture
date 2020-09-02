# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from typing import List
from kapture.io.records import TransferAction, import_record_data_from_dir_auto
from kapture.utils.logging import getLogger


def merge_records_data(image_list: List[List[str]],
                       image_paths: List[str],
                       kapture_path: str,
                       images_import_method: TransferAction):
    """
    Merge several records data. keep only the first image.

    :param image_list: list of image_names
    :param image_paths: list of image paths
    :param kapture_path: root path to the merged kapture
    :param images_import_method: choose how to import actual image files
    """
    assert len(image_list) > 0
    assert len(image_list) == len(image_paths)

    added_images = set()
    for images_filenames, record_dirpath in zip(image_list, image_paths):
        images_filenames_to_add = {images_filename
                                   for images_filename in images_filenames
                                   if images_filename not in added_images}
        import_record_data_from_dir_auto(record_dirpath, kapture_path, images_filenames_to_add, images_import_method)
        diff = set(images_filenames).difference(images_filenames_to_add)
        if len(diff) > 0:
            getLogger().warning(f'Cannot import some images because they were already added: {diff}')
        added_images.update(images_filenames_to_add)
