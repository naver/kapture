#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
import os
import pathlib
from typing import List, Tuple, Union
import numpy as np
from tqdm import tqdm

import path_to_kapture
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_to_file
from kapture.io.csv import ImageFeatureConfig
from kapture.io.features import image_global_features_from_file, global_features_to_filepaths

logger = logging.getLogger('compute_image_pairs')


def stack_global_features(global_features_config: ImageFeatureConfig,
                          global_features_paths: List[Tuple[str, str]]
                          ) -> Tuple[Union[np.array, List[str]], np.ndarray]:
    """
    loads global features and store them inside a numpy array of shape (number_of_images, dsize)

    :param global_features_config: content of global_features.txt, required to load the global features
    :type global_features_config: ImageFeatureConfig
    :param global_features_paths: list of every image and the full path to the corresponding global feature
    :type global_features_paths: List[Tuple[str, str]]
    """
    logger.debug('loading global features')
    number_of_images = len(global_features_paths)

    stacked_features_index = np.empty((number_of_images,), dtype=object)
    stacked_features = np.empty((number_of_images, global_features_config.dsize), dtype=global_features_config.dtype)

    hide_progress_bar = logger.getEffectiveLevel() > logging.INFO
    for i, (image_path, global_feature_path) in tqdm(enumerate(global_features_paths), disable=hide_progress_bar):
        stacked_features_index[i] = image_path
        global_feature = image_global_features_from_file(global_feature_path,
                                                         global_features_config.dtype,
                                                         global_features_config.dsize)
        global_feature = global_feature / np.linalg.norm(global_feature)
        stacked_features[i] = global_feature

    return stacked_features_index, stacked_features


def compute_image_pairs(mapping_path: str,
                        query_path: str,
                        output_path: str,
                        topk: int):
    """
    compute image pairs between query -> mapping from global features, and write the result in a text file

    :param mapping_path: input path to kapture input root directory
    :type mapping_path: str
    :param query_path: input path to a kapture root directory
    :type query_path: str
    :param output_path: output path to pairsfile
    :type output_path: str
    :param topk: the max number of top retained images
    :type topk: int
    """
    logger.info(f'compute_image_pairs. loading mapping: {mapping_path}')
    kdata_mapping = kapture_from_dir(mapping_path)
    assert kdata_mapping.sensors is not None
    assert kdata_mapping.records_camera is not None
    assert kdata_mapping.global_features is not None

    if mapping_path == query_path:
        kdata_query = kdata_mapping
    else:
        logger.info(f'compute_image_pairs. loading query: {query_path}')
        kdata_query = kapture_from_dir(query_path)
        assert kdata_query.sensors is not None
        assert kdata_query.records_camera is not None
        assert kdata_query.global_features is not None

    assert kdata_mapping.global_features is not None
    assert kdata_query.global_features is not None
    assert kdata_mapping.global_features.type_name == kdata_query.global_features.type_name
    assert kdata_mapping.global_features.dtype == kdata_query.global_features.dtype
    assert kdata_mapping.global_features.dsize == kdata_query.global_features.dsize
    global_features_config = ImageFeatureConfig(kdata_mapping.global_features.type_name,
                                                kdata_mapping.global_features.dtype,
                                                kdata_mapping.global_features.dsize)

    logger.info(f'computing pairs from with {kdata_mapping.global_features.type_name}...')

    mapping_global_features_to_filepaths = global_features_to_filepaths(kdata_mapping.global_features,
                                                                        mapping_path)
    mapping_list = list(kapture.flatten(mapping_global_features_to_filepaths, is_sorted=True))
    mapping_indexes, mapping_features = stack_global_features(global_features_config,
                                                              mapping_list)

    if mapping_path == query_path:
        query_indexes, query_features = mapping_indexes, mapping_features
    else:
        query_global_features_to_filepaths = global_features_to_filepaths(kdata_query.global_features,
                                                                          query_path)
        query_list = list(kapture.flatten(query_global_features_to_filepaths, is_sorted=True))
        query_indexes, query_features = stack_global_features(global_features_config,
                                                              query_list)
    # compute similarity matrix
    similarity_matrix = query_features.dot(mapping_features.T)

    # convert similarity matrix to dictionary query_name -> sorted (high score first) list [(mapping_name, score), ...]
    similarity_dict = {}
    for i, line in enumerate(similarity_matrix):
        scores = line
        indexes = np.argsort(-scores)
        query_name = query_indexes[i]
        similarity_dict[query_name] = list(zip(mapping_indexes[indexes], scores[indexes]))

    # get list of image pairs
    image_pairs = []
    for query_image_name, images_to_match in sorted(similarity_dict.items()):
        for mapping_image_name, score in images_to_match[:topk]:
            image_pairs.append([query_image_name, mapping_image_name, score])

    logger.info('saving to file  ...')
    p = pathlib.Path(output_path)
    os.makedirs(str(p.parent.resolve()), exist_ok=True)
    with open(output_path, 'w') as fid:
        table_to_file(fid, image_pairs, header='# query_image, map_image, score')
    logger.info('all done')


def compute_image_pairs_command_line():
    parser = argparse.ArgumentParser(
        description=('Create image pairs files from global features. '
                     'Pairs are computed between query <-> mapping or mapping <-> mapping'))
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--mapping', required=True,
                        help=('input path to kapture input root directory\n'
                              'it must contain global features for all images'))
    parser.add_argument('--query', required=True,
                        help=('input path to a kapture root directory containing query images\n'
                              'it must contain global features for all images\n'
                              'use the same value as mapping if you want to compute mapping <-> mapping matches'))
    parser.add_argument('-o', '--output', required=True,
                        help='output path to pairsfile')

    parser.add_argument('--topk',
                        default=20,
                        type=int,
                        help='the max number of top retained images')

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_image_pairs(args.mapping, args.query, args.output, args.topk)


if __name__ == '__main__':
    compute_image_pairs_command_line()
