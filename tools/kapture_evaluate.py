#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
Kapture evaluation functions to save results on disk.
"""

import argparse
import logging
import os
import os.path as path
import numpy as np
from statistics import mean, median
from math import isnan
from typing import List, Tuple

import path_to_kapture
import kapture
import kapture.io.csv as csv
from kapture.algo.evaluation import evaluate, fill_bins
import kapture.utils.logging
from kapture.utils.paths import safe_remove_file

RESULTS_FILENAME = "scores"
STATISTICS_FILENAME = "stats"
PLOT_FILENAME = "plot"

logger = logging.getLogger('evaluate')


def write_results_to_file(output_folder: str,
                          labels: List[str],
                          results: List[List[Tuple[str, float, float]]],
                          force: bool) -> None:
    """
    Writes evaluation results to text files. Results and labels must be synchronised.

    :param output_folder: full path of folder to write files in.
    :param labels: labels for the result files
    :param results: results to write
    :param force: Silently overwrite results files if already exists.
    """
    for i in range(0, len(results)):
        label = labels[i]
        result = results[i]
        full_path = path.join(output_folder, RESULTS_FILENAME + '_{}'.format(label) + '.txt')
        safe_remove_file(full_path, force)
        results_as_lines = ['{} {} {}\n'.format(name, position_error, rotation_error)
                            for name, position_error, rotation_error in result]
        with open(full_path, 'w') as fid:
            fid.writelines(results_as_lines)


def write_statistics_to_file(output_folder: str,
                             labels: List[str],
                             results: List[List[Tuple[str, float, float]]],
                             bins_as_str: List[str],
                             force: bool) -> None:
    """
    Writes evaluation statistics to text files. Results and labels must be synchronised.

    :param output_folder: full path of folder to write statistics files in.
    :param labels: labels for the statistics files
    :param results: results to compute statistics
    :param bins_as_str: list of bin names
    :param force: Silently overwrite results files if already exists.
    """
    full_path = path.join(output_folder, STATISTICS_FILENAME + '.txt')
    safe_remove_file(full_path, force)

    bins = [(float(split_bin[0]), float(split_bin[1])) for split_bin in map(lambda x: x.split(), bins_as_str)]
    print_line = ''
    for i in range(0, len(results)):
        label = labels[i]
        result = results[i]
        number_of_images = len(result)

        positions_errors_all = [position_error if not isnan(position_error) else float("inf")
                                for name, position_error, rotation_error in result]
        rotation_errors_all = [rotation_error if not isnan(rotation_error) else float("inf")
                               for name, position_error, rotation_error in result]

        positions_errors = [position_error
                            for name, position_error, rotation_error in result if not isnan(position_error)]
        rotation_errors = [rotation_error
                           for name, position_error, rotation_error in result if not isnan(rotation_error)]

        print_line += 'Model: {}\n\n'.format(label)
        print_line += 'Found {} / {} image positions ({:.2f} %).\n'.format(
            len(positions_errors), number_of_images, float(100.0*len(positions_errors)/number_of_images))
        print_line += 'Found {} / {} image rotations ({:.2f} %).\n'.format(
            len(rotation_errors), number_of_images, float(100.0*len(rotation_errors)/number_of_images))

        print_line += 'Localized images: mean=({:.4f}m, {:.4f} deg) / median=({:.4f}m, {:.4f} deg)\n'.format(
            mean(positions_errors),
            mean(rotation_errors),
            median(positions_errors),
            median(rotation_errors))
        print_line += 'All: median=({:.4f}m, {:.4f} deg)\n'.format(median(positions_errors_all),
                                                                   median(rotation_errors_all))
        print_line += 'Min: {:.4f}m; {:.4f} deg\n'.format(min(positions_errors), min(rotation_errors))
        print_line += 'Max: {:.4f}m; {:.4f} deg\n\n'.format(max(positions_errors), max(rotation_errors))

        filled_bins = fill_bins(result, bins)
        bins_lines = ['({}m, {} deg): {:.2f}%\n'.format(
            position_error,
            rotation_error,
            (number_of_images_in_bin / number_of_images) * 100.0)
            for position_error, rotation_error, number_of_images_in_bin in filled_bins]
        print_line += ''.join(bins_lines)
        print_line += '\n'
    print(print_line)
    with open(full_path, 'w') as fid:
        fid.write(print_line)


def plot_localized_over_position_threshold(output_folder: str,
                                           labels: List[str],
                                           results: List[List[Tuple[str, float, float]]],
                                           rotation_threshold: float, plot_max: int,
                                           title: str,
                                           plot_loc: str, plot_font_size: int, plot_legend_font_size: int,
                                           force: bool) -> None:
    """
    Plot image with localized positions.

    :param output_folder: full path of folder to write plot files in.
    :param labels: labels for the plot images
    :param results: results to compute the images to plot
    :param rotation_threshold: rotation threshold
    :param plot_max: max points to plot
    :param title: title for the plotted image
    :param plot_loc: location for the legend
    :param plot_font_size: font size to plot with
    :param plot_legend_font_size: font size for the legend
    :param force: Silently overwrite plot files if already exists.
    """
    import matplotlib.pylab as plt
    plt.rcParams['font.size'] = plot_font_size
    plt.rcParams['legend.fontsize'] = plot_legend_font_size
    position_thresholds = np.linspace(0.0, 1.0, num=101, dtype=np.float64) * plot_max  # thresholds in cm
    bins = [(position_threshold / 100.0, rotation_threshold)
            for position_threshold in position_thresholds]  # convert to thresholds in m
    number_of_images = None
    for i in range(0, len(results)):
        label = labels[i]
        result = results[i]
        if number_of_images is None:
            number_of_images = len(result)
        else:
            # just make sure we are comparing comparable things
            assert(number_of_images == len(result))
        filled_bins = [(t[2] / number_of_images) * 100.0 for t in fill_bins(result, bins)]  # convert back to cm
        plt.plot(position_thresholds, filled_bins, lw=2, label=label)
    plt.xlabel('position error threshold (cm)')
    plt.ylabel('localized images (%)')
    plt.ylim([0, 100])  # 0 to 100 %
    plt.legend(loc=plot_loc)
    plt.xlim([position_thresholds[0], position_thresholds[-1]])
    # plot grid
    grid_step = int(plot_max / 10.0)
    for i in range(grid_step, plot_max - grid_step + 1, grid_step):
        plt.plot([i, i], [0, 100], 'k', alpha=0.1)
    for i in range(10, 91, 10):
        plt.plot([0, plot_max], [i, i], 'k', alpha=0.1)

    plot_title = title
    if rotation_threshold >= 0:
        plot_title += (' ' if title else '') + r'$\alpha_{th}$' + '={} deg'.format(rotation_threshold)
    plt.title(plot_title)

    full_path = path.join(output_folder, PLOT_FILENAME + '.png')
    safe_remove_file(full_path, force)
    plt.savefig(full_path, bbox_inches='tight')


def save_evaluation(results: List[List[Tuple[str, float, float]]],
                    output_dir: str,
                    labels: List[str],
                    bins_as_str: List[str],
                    rotation_threshold: float,
                    plot_max: int,
                    plot_title: str,
                    plot_loc: str, plot_font_size: int, plot_legend_font_size: int,
                    force: bool) -> None:
    """
    Save the evaluation results into results, statistics and plot files. Results and labels must be synchronised.

    :param results: results to save
    :param output_dir: full path of folder to write statistics files in.
    :param labels: labels of the results.
    :param bins_as_str: list of bin names
    :param rotation_threshold: rotation threshold for the plot
    :param plot_max: max points to plot
    :param plot_title: title for the plot
    :param plot_loc: location of the plot legend
    :param plot_font_size: font size to plot with
    :param plot_legend_font_size: font size for the legend
    :param force: Silently overwrite files if already exists.
    """
    write_results_to_file(output_dir, labels, results, force)
    write_statistics_to_file(output_dir, labels, results, bins_as_str, force)
    plot_localized_over_position_threshold(output_dir, labels, results,
                                           rotation_threshold, plot_max, plot_title,
                                           plot_loc, plot_font_size, plot_legend_font_size,
                                           force)


def evaluate_command_line() -> None:
    """
    Do the evaluation using the parameters given on the command line.
    """
    parser = argparse.ArgumentParser(description='Evaluation script for kapture data.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-i', '--inputs', nargs='+',
                        help='input path to kapture data root directory. You can compare multiple models')
    parser.add_argument('--labels', nargs='+', default=[],
                        help='labels for inputs. must be of same length as inputs')
    parser.add_argument('-gt', '--ground-truth', required=True,
                        help='input path to data ground truth root directory in kapture format')
    parser.add_argument('-o', '--output',
                        help='output directory.', required=True)
    parser.add_argument('-l', '--image-list', default="",
                        help='optional, path to a text file containing the list of images to consider'
                        ' (1 line per image). if not present, all gt images are used')
    parser.add_argument('--bins', nargs='+', default=["0.25 2", "0.5 5", "5 10"],
                        help='the desired positions/rotations thresholds for bins'
                        'format is string : position_threshold_in_m space rotation_threshold_in_degree')
    parser.add_argument('-p', '--plot-rotation-threshold', default=-1, type=float,
                        help='rotation threshold for position error threshold plot. negative values -> ignore rotation')
    parser.add_argument('--plot-max', default=100, type=int, help='maximum distance in cm shown in plot')
    parser.add_argument('--plot-title', default="",
                        help='title for position error threshold plot')
    parser.add_argument('--plot-loc', default="best", choices=['best', 'upper right', 'upper left', 'lower left',
                                                               'lower right', 'right', 'center left', 'center right',
                                                               'lower center', 'upper center', 'center'],
                        help='position of plot legend. loc param for plt.legend.')
    parser.add_argument('--plot-font-size', default=15, type=int,
                        help='value for plt.rcParams[\'font.size\']')
    parser.add_argument('--plot-legend-font-size', default=8, type=int,
                        help='value for plt.rcParams[\'legend.fontsize\']')
    parser.add_argument('-f', '-y', '--force', action='store_true', default=False,
                        help='Force delete output directory if already exists')
    args = parser.parse_args()

    logger.setLevel(args.verbose)
    if args.verbose <= logging.DEBUG:
        # also let kapture express its logs
        kapture.utils.logging.getLogger().setLevel(args.verbose)
    assert(len(args.inputs) > 0)
    if len(args.labels) == 0:
        args.labels = [f'input{i}' for i in range(1, len(args.inputs) + 1)]
    assert(len(args.labels) == len(args.inputs))

    try:
        logger.debug(''.join(['\n\t{:13} = {}'.format(k, v)
                              for k, v in vars(args).items()]))

        os.makedirs(args.output, exist_ok=True)

        logger.debug('loading: {}'.format(args.inputs))
        all_kapture_to_eval = [csv.kapture_from_dir(folder) for folder in args.inputs]

        logger.info('loading ground truth data')
        gt_kapture = csv.kapture_from_dir(args.ground_truth)
        assert gt_kapture.records_camera is not None

        if args.image_list:
            with open(args.image_list, 'r') as fid:
                lines = fid.readlines()
                image_set = {line.rstrip() for line in lines}
        else:
            image_set = set(image_name for _, _, image_name in kapture.flatten(gt_kapture.records_camera))
        if len(image_set) == 0:
            logger.info('image_set is empty, for some reason, I could not find images to evaluate')
            exit(0)

        results = [evaluate(kapture_to_eval, gt_kapture, image_set) for kapture_to_eval in all_kapture_to_eval]

        save_evaluation(results, args.output, args.labels, args.bins,
                        args.plot_rotation_threshold, args.plot_max, args.plot_title,
                        args.plot_loc, args.plot_font_size, args.plot_legend_font_size,
                        args.force)

    except Exception as e:
        logger.critical(e)
        if args.verbose > 1:
            raise


if __name__ == '__main__':
    evaluate_command_line()
