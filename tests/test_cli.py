#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Tests of the command line of every kapture tool: parsers build, legacy flags still parse.
"""

import importlib
import logging
import pkgutil
import tomllib
import unittest
from os import path

from jsonargparse import capture_parser

import kapture.tools
from kapture.io.records import TransferAction
from kapture.utils.cli import to_logging_level, setup_logging


def _entry_points() -> dict:
    """ {entry point name: module path} from pyproject.toml """
    pyproject_path = path.join(path.dirname(__file__), '..', 'pyproject.toml')
    with open(pyproject_path, 'rb') as f:
        scripts = tomllib.load(f)['project']['scripts']
    return {name: target.split(':') for name, target in scripts.items()}


def _parser(tool_module_name: str):
    module = importlib.import_module(f'kapture.tools.{tool_module_name}')
    return capture_parser(module.main)


class TestEntryPoints(unittest.TestCase):
    def test_every_tool_is_an_entry_point_with_main(self):
        tools = {m.name for m in pkgutil.iter_modules(kapture.tools.__path__)}
        entry_points = _entry_points()
        self.assertEqual(tools, {module.split('.')[-1] for module, _ in entry_points.values()})
        for name, (module, function) in entry_points.items():
            self.assertEqual(function, 'main', f'{name} should point to main')

    def test_every_parser_builds_with_verbosity_flags(self):
        for name, (module, _) in _entry_points().items():
            with self.subTest(tool=name):
                try:
                    parser = _parser(module.split('.')[-1])
                except ModuleNotFoundError as e:  # optional dependency (eg. pyzed)
                    self.skipTest(f'optional dependency missing: {e}')
                option_strings = {opt for action in parser._actions for opt in action.option_strings}
                self.assertIn('-v', option_strings)
                self.assertIn('--verbose', option_strings)
                self.assertIn('-q', option_strings)
                self.assertIn('--config', option_strings)
                self.assertNotIn('--no_verbose', option_strings)


class TestLegacyFlags(unittest.TestCase):
    def test_verbosity_forms(self):
        parser = _parser('kapture_print')
        self.assertEqual(parser.parse_args(['-i', 'k']).verbose, 'warning')
        self.assertEqual(parser.parse_args(['-i', 'k', '-v']).verbose, 'info')
        self.assertEqual(parser.parse_args(['-i', 'k', '-v', 'debug']).verbose, 'debug')
        self.assertEqual(parser.parse_args(['-i', 'k', '-v', '10']).verbose, 10)
        self.assertTrue(parser.parse_args(['-i', 'k', '-q']).quiet)
        self.assertTrue(parser.parse_args(['-i', 'k', '--silent']).quiet)

    def test_print_aliases(self):
        parser = _parser('kapture_print')
        args = parser.parse_args(['-k', 'kapt', '-a', '-d', '-t', 'second', '-f', '%H'])
        self.assertEqual(args.input, 'kapt')
        self.assertTrue(args.show_all)
        self.assertTrue(args.show_detail)
        self.assertEqual(args.timestamp_unit, 'second')
        self.assertEqual(args.timestamp_formatting, '%H')
        args = parser.parse_args(['--kapture', 'kapt', '--all', '--detail'])
        self.assertEqual((args.input, args.show_all, args.show_detail), ('kapt', True, True))

    def test_merge_lists_and_booleans(self):
        parser = _parser('kapture_merge')
        args = parser.parse_args(['-y', '-i', 'a', 'b', 'c', '-o', 'out', '--keep-sensor-ids',
                                  '-s', 'matches', 'points3d', '--image_transfer', 'copy'])
        self.assertEqual(args.inputs, ['a', 'b', 'c'])
        self.assertEqual(args.output, 'out')
        self.assertTrue(args.force)
        self.assertTrue(args.keep_sensor_ids)
        self.assertEqual(args.skip, ['matches', 'points3d'])
        self.assertEqual(args.image_transfer, TransferAction.copy)
        args = parser.parse_args(['-i', 'a', '-o', 'out'])
        self.assertEqual((args.force, args.keep_sensor_ids, args.skip), (False, False, []))
        with self.assertRaises(SystemExit):
            parser.parse_args(['-i', 'a', '-o', 'out', '-s', 'bogus'])

    def test_colmap_import_legacy_aliases(self):
        parser = _parser('kapture_import_colmap')
        args = parser.parse_args(['-v', '-f', '-db', 'x.db', '-txt', 'rec', '-im', 'imgs', '-rig', 'r.json',
                                  '-kpt', 'R2D2', '-desc', 'R2D2', '-k', 'out', '--no_geometric_filtering'])
        self.assertEqual(args.output, 'out')
        self.assertEqual(args.database, 'x.db')
        self.assertEqual(args.reconstruction, 'rec')
        self.assertEqual(args.images, 'imgs')
        self.assertEqual(args.rig, 'r.json')
        self.assertEqual((args.keypoints_type, args.descriptors_type), ('R2D2', 'R2D2'))
        self.assertFalse(args.geometric_filtering)
        self.assertTrue(args.force)
        args = parser.parse_args(['--kapture', 'out', '--keypoints-type', 'SIFT'])
        self.assertEqual((args.output, args.keypoints_type, args.geometric_filtering), ('out', 'SIFT', True))

    def test_image_folder_config_round_trip(self):
        parser = _parser('kapture_import_image_folder')
        args = parser.parse_args(['-i', 'imgs', '-o', 'kapt', '--image-transfer', 'link_relative', '-y'])
        dumped = parser.dump(args)
        reloaded = parser.parse_string(dumped)
        self.assertEqual(reloaded.input, 'imgs')
        self.assertEqual(reloaded.image_transfer, TransferAction.link_relative)
        self.assertTrue(reloaded.force)


class TestVerbosityHelpers(unittest.TestCase):
    def test_to_logging_level(self):
        self.assertEqual(to_logging_level('debug'), logging.DEBUG)
        self.assertEqual(to_logging_level('INFO'), logging.INFO)
        self.assertEqual(to_logging_level(10), logging.DEBUG)
        self.assertEqual(to_logging_level('10'), logging.DEBUG)
        with self.assertRaises(ValueError):
            to_logging_level('bogus')

    def test_setup_logging(self):
        logger = logging.getLogger('test_cli')
        self.assertEqual(setup_logging(logger, 'info'), logging.INFO)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(setup_logging(logger, 'info', quiet=True), logging.CRITICAL)


if __name__ == '__main__':
    unittest.main()
