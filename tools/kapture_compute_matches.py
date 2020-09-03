#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import argparse
import logging
from functools import lru_cache
from typing import List, Tuple
import torch
from tqdm import tqdm
from collections import OrderedDict

import path_to_kapture  # noqa: F401
import kapture
import kapture.utils.logging
from kapture.io.csv import kapture_from_dir, table_from_file
from kapture.algo.matching import MatchPairNnTorch
from kapture.io.features import get_descriptors_fullpath, get_matches_fullpath
from kapture.io.features import image_descriptors_from_file
from kapture.io.features import matches_check_dir, image_matches_to_file

logger = logging.getLogger('compute_matches')


def get_pairs_from_file(pairsfile_path: str) -> List[Tuple[str, str]]:
    """
    read a pairs file (csv with 3 fields, name1, name2, score) and return the list of matches

    :param pairsfile_path: path to pairsfile
    :type pairsfile_path: str
    """
    logger.info('reading pairs from pairsfile')
    image_pairs = []
    with open(pairsfile_path, 'r') as fid:
        table = table_from_file(fid)
        for query_name, map_name, _ in table:  # last field score is not used
            if query_name != map_name:
                image_pairs.append((query_name, map_name) if query_name < map_name else (map_name, query_name))
    # remove duplicates without breaking order
    image_pairs = list(OrderedDict.fromkeys(image_pairs))
    return image_pairs


@lru_cache(maxsize=50)
def load_descriptors(input_path: str, image_name: str, dtype, dsize):
    """
    load a descriptor. this functions caches up to 50 descriptors

    :param input_path: input path to kapture input root directory
    :param image_name: name of the image
    :param dtype: dtype of the numpy array
    :param dsize: size of the numpy array
    """
    descriptors_path = get_descriptors_fullpath(input_path, image_name)
    return image_descriptors_from_file(descriptors_path, dtype, dsize)


def compute_matches(input_path: str,
                    pairsfile_path: str):
    """
    compute matches from descriptors. images to match are selected from a pairsfile (csv with name1, name2, score)

    :param input_path: input path to kapture input root directory
    :type input_path: str
    :param pairsfile_path: path to pairs file (csv with 3 fields, name1, name2, score)
    :type pairsfile_path: str
    """
    logger.info(f'compute_matches. loading input: {input_path}')
    kdata = kapture_from_dir(input_path)
    assert kdata.sensors is not None
    assert kdata.records_camera is not None
    assert kdata.descriptors is not None

    image_pairs = get_pairs_from_file(pairsfile_path)
    matcher = MatchPairNnTorch(use_cuda=torch.cuda.is_available())
    new_matches = kapture.Matches()

    logger.info('compute_matches. entering main loop...')
    hide_progress_bar = logger.getEffectiveLevel() > logging.INFO
    for image_path1, image_path2 in tqdm(image_pairs, disable=hide_progress_bar):
        if image_path1 == image_path2:
            continue
        if image_path1 > image_path2:
            image_path1, image_path2 = image_path2, image_path1

        if image_path1 not in kdata.descriptors or image_path2 not in kdata.descriptors:
            logger.warning('unable to find descriptors for image pair : '
                           '\n\t{} \n\t{}'.format(image_path1, image_path2))
            continue

        descriptor1 = load_descriptors(input_path, image_path1, kdata.descriptors.dtype, kdata.descriptors.dsize)
        descriptor2 = load_descriptors(input_path, image_path2, kdata.descriptors.dtype, kdata.descriptors.dsize)
        matches = matcher.match_descriptors(descriptor1, descriptor2)
        matches_path = get_matches_fullpath((image_path1, image_path2), input_path)
        image_matches_to_file(matches_path, matches)
        new_matches.add(image_path1, image_path2)

    if not matches_check_dir(new_matches, input_path):
        logger.critical('matching ended successfully but not all files were saved')
    logger.info('all done')


def compute_matches_command_line():
    parser = argparse.ArgumentParser(
        description='Compute matches with nearest neighbors from descriptors.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument('-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
                                  action=kapture.utils.logging.VerbosityParser,
                                  help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument('-q', '--silent', '--quiet',
                                  action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--input', required=True,
                        help=('input path to kapture input root directory\n'
                              'it must contain all images (query + train) and their local features'))
    parser.add_argument('--pairsfile-path',
                        required=True,
                        type=str,
                        help=('text file in the csv format; where each line is image_name1, image_name2, score '
                              'which contains the image pairs to match'))

    args = parser.parse_args()
    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)

    logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                          for k, v in vars(args).items()]))
    compute_matches(args.input, args.pairsfile_path)


if __name__ == '__main__':
    compute_matches_command_line()
