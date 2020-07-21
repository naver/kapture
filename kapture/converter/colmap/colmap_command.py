# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Colmap command execution in sub processes
"""

import logging
import os
import os.path as path
import subprocess
from typing import List, Optional

import kapture
from kapture.utils.paths import safe_remove_file
from .database_extra import save_match_list

logger = logging.getLogger('colmap')


def run_colmap_command(colmap_binary_path: str, args: List[str]) -> None:
    """
    run any colmap command

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param args: list of arguments that will be passed to the colmap command
    :type args: List[str]
    :raises ValueError: colmap subprocess did not return 0
    """
    args.insert(0, colmap_binary_path)
    logger.info(args)
    colmap_process = subprocess.Popen(args)
    colmap_process.wait()

    if colmap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {colmap_process.returncode} )')


def run_feature_extractor(colmap_binary_path: str,
                          colmap_use_cpu: bool,
                          colmap_gpu_index: str,
                          colmap_db_path: str,
                          images_path: str,
                          image_list_path: str,
                          colmap_options: List[str] = None) -> None:
    """
    run colmap feature_extractor:
     Perform feature extraction or import features for a set of images

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_use_cpu: add --SiftExtraction.use_gpu 0
    :type colmap_use_cpu: bool
    :param colmap_gpu_index: add --SiftExtraction.gpu_index {colmap_gpu_index}
    :type colmap_gpu_index: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param images_path: value for --image_path
    :type images_path: str
    :param image_list_path: value for --image_list_path
    :type image_list_path: str
    :param colmap_options: list of additional parameters to add to the command, defaults to None
    :type colmap_options: List[str], optional
    """
    feature_args = ["feature_extractor",
                    "--database_path",
                    colmap_db_path,
                    "--image_path",
                    images_path,
                    "--image_list_path",
                    image_list_path]
    if colmap_options is not None and len(colmap_options) > 0:
        feature_args += colmap_options
    if colmap_use_cpu:
        feature_args += [
            "--SiftExtraction.use_gpu",
            "0"
        ]
    elif colmap_gpu_index:
        feature_args += [
            "--SiftExtraction.gpu_index",
            colmap_gpu_index
        ]

    run_colmap_command(colmap_binary_path, feature_args)


def run_vocab_tree_matcher(colmap_binary_path: str,
                           colmap_use_cpu: bool,
                           colmap_gpu_index: str,
                           colmap_db_path: str,
                           vocab_tree_path: str,
                           images_path: str = "") -> None:
    """
    run colmap vocab_tree_matcher:
     Perform feature matching after performing feature extraction

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_use_cpu: add --SiftExtraction.use_gpu 0
    :type colmap_use_cpu: bool
    :param colmap_gpu_index: add --SiftExtraction.gpu_index {colmap_gpu_index}
    :type colmap_gpu_index: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param vocab_tree_path: value for --VocabTreeMatching.vocab_tree_path
    :type vocab_tree_path: str
    :param images_path: value for --VocabTreeMatching.match_list_path, defaults to ""
    :type images_path: str, optional
    """
    vocab_tree_matcher_args = ["vocab_tree_matcher",
                               "--database_path",
                               colmap_db_path,
                               "--VocabTreeMatching.vocab_tree_path",
                               vocab_tree_path]
    if images_path != "":
        vocab_tree_matcher_args += ["--VocabTreeMatching.match_list_path", images_path]

    if colmap_use_cpu:
        vocab_tree_matcher_args += [
            "--SiftMatching.use_gpu",
            "0"
        ]
    elif colmap_gpu_index:
        vocab_tree_matcher_args += [
            "--SiftMatching.gpu_index",
            colmap_gpu_index
        ]
    run_colmap_command(colmap_binary_path, vocab_tree_matcher_args)


def run_matches_importer(colmap_binary_path: str,
                         colmap_use_cpu: bool,
                         colmap_gpu_index: Optional[str],
                         colmap_db_path: str,
                         match_list_path: str) -> None:
    """
    run colmap matches_importer:
     Perform geometric verification on matches

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_use_cpu: add --SiftExtraction.use_gpu 0
    :type colmap_use_cpu: bool
    :param colmap_gpu_index: add --SiftExtraction.gpu_index {colmap_gpu_index}
    :type colmap_gpu_index: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param match_list_path: value for --match_list_path
    :type match_list_path: str
    """
    matches_importer_args = ["matches_importer",
                             "--database_path",
                             colmap_db_path,
                             "--match_list_path",
                             match_list_path,
                             "--match_type",
                             'pairs']
    if colmap_use_cpu:
        matches_importer_args += [
            "--SiftMatching.use_gpu",
            "0"
        ]
    elif colmap_gpu_index:
        matches_importer_args += [
            "--SiftMatching.gpu_index",
            colmap_gpu_index
        ]
    run_colmap_command(colmap_binary_path, matches_importer_args)


def run_matches_importer_from_kapture(colmap_binary_path: str,
                                      colmap_use_cpu: bool,
                                      colmap_gpu_index: Optional[str],
                                      colmap_db_path: str,
                                      kapture_data: kapture.Kapture,
                                      force: bool = True,
                                      clean: bool = True) -> None:
    """
    export list of matches from kapture data then run colmap matches_importer

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_use_cpu: add --SiftExtraction.use_gpu 0
    :type colmap_use_cpu: bool
    :param colmap_gpu_index: add --SiftExtraction.gpu_index {colmap_gpu_index}
    :type colmap_gpu_index: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param kapture_data: kapture data that contains the matches (that are already in the colmap database) to verify
    :type kapture_data: kapture.Kapture
    :param force: do not ask before overwriting match_list.txt, defaults to True
    :type force: bool, optional
    :param clean: remove match_list.txt before exiting, defaults to True
    :type clean: bool, optional
    """
    db_dir = path.dirname(colmap_db_path)
    match_list_path = path.join(db_dir, 'match_list.txt')
    safe_remove_file(match_list_path, force)
    save_match_list(kapture_data, match_list_path)
    run_matches_importer(colmap_binary_path, colmap_use_cpu, colmap_gpu_index, colmap_db_path, match_list_path)
    if clean:
        os.remove(match_list_path)


def run_point_triangulator(colmap_binary_path: str,
                           colmap_db_path: str,
                           images_path: str,
                           input_path: str,
                           output_path: str,
                           point_triangulator_options: List[str]) -> None:
    """
    run colmap point_triangulator:
     Triangulate all observations of registered images in an existing model using the feature matches in a database

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param images_path: value for --image_path
    :type images_path: str
    :param input_path: value for --input_path
    :type input_path: str
    :param output_path: value for --output_path
    :type output_path: str
    :param point_triangulator_options: list of additional parameters to add to the command
    :type point_triangulator_options: List[str]
    """
    point_triangulator_args = ["point_triangulator",
                               "--database_path",
                               colmap_db_path,
                               "--image_path",
                               images_path,
                               "--input_path",
                               input_path,
                               "--output_path",
                               output_path]
    if point_triangulator_options is not None and len(point_triangulator_options) > 0:
        point_triangulator_args += point_triangulator_options
    run_colmap_command(colmap_binary_path, point_triangulator_args)


def run_mapper(colmap_binary_path: str,
               colmap_db_path: str,
               images_path: str,
               input_path: Optional[str],
               output_path: str,
               mapper_options: List[str]) -> None:
    """
    run colmap mapper:
     Sparse 3D reconstruction / mapping of the dataset using SfM after performing feature extraction and matching

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param images_path: value for --image_path
    :type images_path: str
    :param input_path: value for --input_path
    :type input_path: Optional[str]
    :param output_path: value for --output_path
    :type output_path: str
    :param mapper_options: list of additional parameters to add to the command
    :type mapper_options: List[str]
    """
    mapper_args = ["mapper",
                   "--database_path",
                   colmap_db_path,
                   "--image_path",
                   images_path,
                   "--output_path",
                   output_path]
    if input_path is not None:
        mapper_args += [
            "--input_path",
            input_path]
    if mapper_options is not None and len(mapper_options) > 0:
        mapper_args += mapper_options
    run_colmap_command(colmap_binary_path, mapper_args)


def run_bundle_adjustment(colmap_binary_path: str,
                          input_path: str,
                          output_path: str,
                          output_rig_path: str = "") -> None:
    """
    run colmap bundle_adjuster or colmap rig_bundle_adjuster (if output_rig_path is provided)

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param input_path: value for --input_path
    :type input_path: str
    :param output_path: value for --output_path
    :type output_path: str
    :param output_rig_path: value for --rig_config_path, if set, run rig_bundle_adjuster instead of bundle_adjuster
    :type output_rig_path: str, optional
    """
    if output_rig_path:
        logging.info("Run bundle adjuster with rig")
        args = ["rig_bundle_adjuster", "--rig_config_path", output_rig_path]
    else:
        args = ["bundle_adjuster"]
    args.extend(["--input_path", input_path,
                 "--output_path", output_path])
    run_colmap_command(colmap_binary_path, args)


def run_image_registrator(colmap_binary_path: str,
                          colmap_db_path: str,
                          input_path: str,
                          output_path: str,
                          image_registrator_options: List[str]) -> None:
    """
    run colmap image_registrator:
     Register new images in the database against an existing model

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param colmap_db_path: value for --database_path
    :type colmap_db_path: str
    :param input_path: value for --input_path
    :type input_path: str
    :param output_path: value for --output_path
    :type output_path: str
    :param image_registrator_options: list of additional parameters to add to the command
    :type image_registrator_options: List[str]
    """
    image_registrator_args = ["image_registrator",
                              "--database_path",
                              colmap_db_path,
                              "--input_path",
                              input_path,
                              "--output_path",
                              output_path]
    if image_registrator_options is not None and len(image_registrator_options) > 0:
        image_registrator_args += image_registrator_options
    run_colmap_command(colmap_binary_path, image_registrator_args)


def run_model_converter(colmap_binary_path: str,
                        input_path: str,
                        output_path: str) -> None:
    """
    run colmap model_converter with --output_type TXT:
     convert reconstruction from binary files to TXT files

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param input_path: value for --input_path
    :type input_path: str
    :param output_path: value for --output_path
    :type output_path: str
    """
    model_converter_args = ["model_converter",
                            "--input_path",
                            input_path,
                            "--output_path",
                            output_path,
                            "--output_type",
                            "TXT"]
    run_colmap_command(colmap_binary_path, model_converter_args)


def run_image_undistorter(colmap_binary_path: str,
                          image_path: str,
                          input_path: str,
                          output_path: str) -> None:
    """
    run colmap image_undistorter:
     Undistort images and/or export them for MVS or to external dense reconstruction software, such as CMVS/PMVS

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param image_path: value for -image_path
    :type image_path: str
    :param input_path: value for --input_path
    :type input_path: str
    :param output_path: value for --output_path
    :type output_path: str
    """
    image_undistorter_args = ["image_undistorter",
                              "--image_path",
                              image_path,
                              "--input_path",
                              input_path,
                              "--output_path",
                              output_path]
    run_colmap_command(colmap_binary_path, image_undistorter_args)


def run_patch_match_stereo(colmap_binary_path: str,
                           workspace_path: str) -> None:
    """
    run colmap patch_match_stereo:
     Dense 3D reconstruction / mapping using MVS after running the image_undistorter to initialize the workspace

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param workspace_path: value for --workspace_path
    :type workspace_path: str
    """
    patch_match_stereo_args = ["patch_match_stereo",
                               "--workspace_path",
                               workspace_path]
    run_colmap_command(colmap_binary_path, patch_match_stereo_args)


def run_stereo_fusion(colmap_binary_path: str,
                      workspace_path: str,
                      output_path: str) -> None:
    """
    run colmap stereo_fusion

    :param colmap_binary_path: path to colmap executable
    :type colmap_binary_path: str
    :param workspace_path: value for --workspace_path
    :type workspace_path: str
    :param output_path: value for --output_path
    :type output_path: str
    """
    stereo_fusion_args = ["stereo_fusion",
                          "--workspace_path",
                          workspace_path,
                          "--output_path",
                          output_path]
    run_colmap_command(colmap_binary_path, stereo_fusion_args)
