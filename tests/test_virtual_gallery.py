#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import unittest
import os.path as path
import tempfile
# kapture
import path_to_kapture  # enables import kapture  # noqa: F401
from kapture.io.csv import kapture_from_dir
from kapture.algo.compare import equal_kapture
from kapture.converter.virtual_gallery.import_virtual_gallery import import_virtual_gallery  # noqa: E402
from kapture.core.Trajectories import rigs_remove_inplace
from kapture.io.records import TransferAction


class TestImportVirtualGallery(unittest.TestCase):

    def setUp(self):
        samples_folder = path.abspath(path.join(path.dirname(__file__),  '../samples/'))
        self.virtual_gallery_folder = path.join(samples_folder, 'virtual_gallery')
        self.virtual_gallery_1_0_0_folder = path.join(self.virtual_gallery_folder, '1.0.0')

    def test_import(self):
        image_transfer_action = TransferAction.skip
        with tempfile.TemporaryDirectory() as tmpdirname:
            configuration = 'all'
            light_range = [1, 4]
            loop_range = [1]
            camera_range = list(range(0, 6))
            occlusion_range = [2, 3]
            as_rig = False

            import_virtual_gallery(self.virtual_gallery_1_0_0_folder, configuration,
                                   light_range, loop_range, camera_range, occlusion_range, as_rig,
                                   image_transfer_action,
                                   tmpdirname, force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.virtual_gallery_folder, 'kapture/all'))
            imported_virtual_gallery_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_virtual_gallery_data, expected_kdata))

        with tempfile.TemporaryDirectory() as tmpdirname:
            configuration = 'all'
            light_range = [4]
            loop_range = [1]
            camera_range = [2]
            occlusion_range = [3]
            as_rig = False

            import_virtual_gallery(self.virtual_gallery_1_0_0_folder, configuration,
                                   light_range, loop_range, camera_range, occlusion_range, as_rig,
                                   image_transfer_action,
                                   tmpdirname, force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.virtual_gallery_folder, 'kapture/reduced'))
            imported_virtual_gallery_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_virtual_gallery_data, expected_kdata))

        with tempfile.TemporaryDirectory() as tmpdirname:
            configuration = 'training'
            light_range = [1, 4]
            loop_range = [1]
            camera_range = list(range(0, 6))
            occlusion_range = [2, 3]
            as_rig = False

            import_virtual_gallery(self.virtual_gallery_1_0_0_folder, configuration,
                                   light_range, loop_range, camera_range, occlusion_range, as_rig,
                                   image_transfer_action,
                                   tmpdirname, force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.virtual_gallery_folder, 'kapture/training'))
            imported_virtual_gallery_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_virtual_gallery_data, expected_kdata))

        with tempfile.TemporaryDirectory() as tmpdirname:
            configuration = 'training'
            light_range = [1, 4]
            loop_range = [1]
            camera_range = list(range(0, 6))
            occlusion_range = [2, 3]
            as_rig = True

            import_virtual_gallery(self.virtual_gallery_1_0_0_folder, configuration,
                                   light_range, loop_range, camera_range, occlusion_range, as_rig,
                                   image_transfer_action,
                                   tmpdirname, force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.virtual_gallery_folder, 'kapture/training'))
            imported_virtual_gallery_data = kapture_from_dir(tmpdirname)
            self.assertFalse(equal_kapture(imported_virtual_gallery_data, expected_kdata))

            rigs_remove_inplace(imported_virtual_gallery_data.trajectories, imported_virtual_gallery_data.rigs)
            self.assertTrue(equal_kapture(imported_virtual_gallery_data, expected_kdata))

        with tempfile.TemporaryDirectory() as tmpdirname:
            configuration = 'testing'
            light_range = [1, 4]
            loop_range = [1]
            camera_range = list(range(0, 6))
            occlusion_range = [2, 3]
            as_rig = False

            import_virtual_gallery(self.virtual_gallery_1_0_0_folder, configuration,
                                   light_range, loop_range, camera_range, occlusion_range, as_rig,
                                   image_transfer_action,
                                   tmpdirname, force_overwrite_existing=True)

            expected_kdata = kapture_from_dir(path.join(self.virtual_gallery_folder, 'kapture/testing'))
            imported_virtual_gallery_data = kapture_from_dir(tmpdirname)
            self.assertTrue(equal_kapture(imported_virtual_gallery_data, expected_kdata))


if __name__ == '__main__':
    unittest.main()
