#!/usr/bin/env python3
# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

"""
script to easily  download kapture datasets.
"""

import argparse
import logging
import sys
import os
import os.path as path
import requests
import yaml
import fnmatch
from typing import Dict, Optional, List, Set
from tqdm import tqdm
import textwrap
import path_to_kapture  # noqa: F401
# import kapture
import kapture.utils.logging
from kapture.converter.downloader.download import download_file, get_remote_file_size
from kapture.converter.downloader.archives import untar_file, compute_sha256sum
from kapture.io.csv import get_version_from_csv_file
from kapture.utils.upgrade import upgrade_1_0_to_1_1_inplace, upgrade_1_0_to_1_1_orphan_features

logger = logging.getLogger('downloader')
logging.basicConfig(format='%(levelname)-8s::%(name)s: %(message)s')

INDEX_FILENAME = 'kapture_dataset_index.yaml'
INSTALL_LOG_FILENAME = 'kapture_dataset_installed.yaml'
LICENSE_LOG_FILENAME = 'kapture_dataset_licenses.yaml'
DEFAULT_DATASET_PATH = path.normpath(path.abspath('.'))
DEFAULT_REPOSITORY_URL = 'https://github.com/naver/kapture/raw/main/dataset'

# absolute path to root directory where all datasets are installed, its set for the all runtime
install_dirpath = None


def ask_confirmation(question):
    """
    ask "question" to the user.
    The "answer" return value is True for "yes" or False for "no".
    """
    validate = ['yes', 'y', 'ye']
    prompt = ' [y/N]\n'
    sys.stdout.write(question + prompt)
    user_choice = input().lower()
    return user_choice in validate


class InstallDir:
    """ Manage access to dataset installation dir and files into  """
    def __init__(
            self,
            index_filepath: str,
            install_dir_path: str):
        self._index_filepath = index_filepath
        self._install_dir_path = install_dir_path
        self._dataset_index_filepath = path.join(install_dir_path, INDEX_FILENAME)
        self._dataset_installed_list_filepath = path.join(install_dir_path, INSTALL_LOG_FILENAME)
        self._license_agreed_list_filepath = path.join(install_dir_path, LICENSE_LOG_FILENAME)

        self._dataset_index = None
        self._installed_index_cache = None  # list of already installed

    @property
    def install_root_path(self) -> str:
        return self._install_dir_path

    def _load_installed_index(self):
        """ load installed index from file. Index is empty if file does not exists. """
        logger.debug(f'loading {self._dataset_installed_list_filepath}')
        installed_index = set()
        if path.isfile(self._dataset_installed_list_filepath):
            with open(self._dataset_installed_list_filepath, 'rt') as f:
                installed_index_yaml = yaml.safe_load(f)
                if installed_index_yaml is not None:
                    installed_index = set(installed_index_yaml)

        self._installed_index_cache = installed_index

    def _write_installed_index(self):
        """ write installed index to file """
        logger.debug(f'writing {self._dataset_installed_list_filepath}')
        with open(self._dataset_installed_list_filepath, 'wt') as f:
            yaml.dump(list(self._installed_index), f)

    # Installed index
    @property
    def _installed_index(self) -> Set[str]:
        if self._installed_index_cache is None:
            # first time, need to be loaded
            self._load_installed_index()
        assert self._installed_index_cache is not None
        return self._installed_index_cache

    def mark_as_installed(self, dataset_name: str, installed: bool = True):
        _ = self._installed_index  # make sure its up to date
        if not self.is_installed(dataset_name) and installed:
            logger.debug(f'dataset marked installed {dataset_name} ')
            self._installed_index_cache.add(dataset_name)
            self._write_installed_index()
        if self.is_installed(dataset_name) and not installed:
            logger.debug(f'dataset marked uninstalled {dataset_name} ')
            self._installed_index_cache.discard(dataset_name)
            self._write_installed_index()

    def is_installed(self, dataset_name: str) -> bool:
        return dataset_name in self._installed_index

    # dataset index
    def load_datasets_from_file(
            self,
            filter_patterns: Optional[List[str]] = None
    ):
        """
        Parses and load data from the index files, under yaml format.
        the yaml file looks like :
        ----
        robotcar_seasons_02:
          url: http://download.europe.naverlabs.com//kapture/robotcar_seasons_02.tar.gz
          sha256sum: 542ef47c00d5e387cfb0dcadb2459ae2fb17d59010cc51bae0c49403b4fa6a18
          license: https://filesamples.com/samples/document/txt/sample3.txt # optional
        ----

        :param filter_patterns: optional input list of unix-like patterns (e.g. SiLDa*) to filter datasets
        :return: dict name -> [url, sub_path, sha256sum]
        """
        if not path.isfile(self._index_filepath):
            raise FileNotFoundError('no index file: do an update.')
        with open(self._index_filepath, 'rt') as f:
            datasets_yaml = yaml.safe_load(f)

        if len(datasets_yaml) == 0:
            raise FileNotFoundError('invalid index file: do an update.')

        nb_total = len(datasets_yaml)
        # filter only dataset matching filter_patterns
        if filter_patterns:
            datasets_yaml = {dataset_name: data_yaml
                             for dataset_name, data_yaml in datasets_yaml.items()
                             if any(fnmatch.fnmatch(dataset_name, pattern) for pattern in filter_patterns)}

        logger.debug(f'will prob status for {len(datasets_yaml)}/{nb_total} datasets ...')
        datasets = {}
        hide_progress_bar = True or logger.getEffectiveLevel() > logging.INFO
        for dataset_name, data_yaml in tqdm(datasets_yaml.items(), disable=hide_progress_bar):
            datasets[dataset_name] = Dataset(name=dataset_name,
                                             install_dir=self,
                                             archive_url=data_yaml['url'],
                                             archive_sha256sum=data_yaml['sha256sum'],
                                             install_script_filename=data_yaml.get('install_script'),
                                             license_url=data_yaml.get('license'))

        return datasets

    def read_licenses(self) -> Set[str]:
        """ returns the list of licences already agreed by user """
        license_list = set()
        if path.isfile(self._license_agreed_list_filepath):
            # licences exists
            with open(self._license_agreed_list_filepath, 'rt') as file:
                license_list_yaml = yaml.safe_load(file)
                if license_list_yaml is not None:
                    license_list = set(license_list_yaml)

        return license_list

    def update_licenses(self, licenses_list_updt: Set[str]):
        """ updates the list licences by adding the givens ones  """
        licenses_list = self.read_licenses()
        licenses_list.update(licenses_list_updt)
        with open(self._license_agreed_list_filepath, 'wt') as file:
            yaml.dump(list(licenses_list), file)


class Dataset:
    def __init__(
            self,
            name: str,
            install_dir: InstallDir,
            archive_url: str,
            archive_sha256sum: str,
            install_script_filename: Optional[str] = None,
            license_url: Optional[str] = None
    ):
        """
        :param name: name of the archive (dataset or part of a dataset)
        :param install_dir: root directory where all datasets are installed.
        :param archive_url: remote url of the dataset archive (tar).
        :param archive_sha256sum: expected sha256 sum of the archive file.
        :param install_script_filename: if given, this script is to be called to finish installation (eg. dl 3rd party).
        :param license_url: if given, url to license file to agree on.
        """
        self._name = name

        self._install_dir = install_dir
        self._archive_filepath = path.join(install_dir.install_root_path, name + '.tar.gz')
        self._archive_url = archive_url
        self._archive_sha256sum_remote = archive_sha256sum
        self._install_script_filename = install_script_filename
        # license
        self._license_url = license_url

    def mark_as_installed(self, installed=True):
        self._install_dir.mark_as_installed(self._name, installed)

    def is_installed(self) -> bool:
        return self._install_dir.is_installed(self._name)

    def is_sha256_consistent(self) -> bool:
        """ check sha256sum of the archive against the expected sha256. Returns true if they are the same. """
        if not path.isfile(self._archive_filepath):
            return False
        # size is consistent, check sha256
        logger.debug(f'checking sha256sum of {path.basename(self._archive_filepath)}, may take a while...')
        sha256sum_archive_local = compute_sha256sum(self._archive_filepath)
        if sha256sum_archive_local != self._archive_sha256sum_remote:
            logger.warning(f'sha256sum discrepancy for {self._archive_filepath} :\n'
                           f'\tlocal :{sha256sum_archive_local}\n'
                           f'\tremote:{self._archive_sha256sum_remote}')
            return False
        logger.debug(f'{path.basename(self._archive_filepath)} is valid (sha256sum)')
        return True

    @property
    def url(self) -> str:
        return self._archive_url

    @property
    def license_url(self) -> Optional[str]:
        return self._license_url

    def prob_status(self, check_online=False) -> str:
        """
        gives the actual dataset status
         - not installed: means is not installed (wo info about the server)
         - online: means not installed, not downloaded, but reachable.
         - not reachable: means not installed, not downloaded, but NOT reachable.
         - installed: means has been downloaded and installed (extracted).
         - downloaded: means has been downloaded (tar) but not installed (extracted) yet.
         - incomplete: means partially downloaded
         - corrupted: means that the downloaded archive is corrupted (inconsistent size or sh256).
         - unknown: should not happen

        :param check_online: If true, will ping the url to check its actually online
        """

        probing_status = None
        size_archive_local = size_archive_online = -1
        if self.is_installed():
            # yaml file says its installed, trust it (no other choices).
            probing_status = 'installed'

        if probing_status is None and not path.isfile(self._archive_filepath):
            # not installed, no archive there, check its online
            if not check_online:
                probing_status = 'not installed'
            else:
                remote_file_size_in_bytes = get_remote_file_size(url=self._archive_url)
                if remote_file_size_in_bytes is not None:
                    probing_status = f'online {remote_file_size_in_bytes / (1024 * 1024):5.0f} MB '
                else:
                    probing_status = 'not reachable'

        # not installed, but archive there, check 1) its incomplete or 2) corrupted. If neither, is just downloaded.
        if probing_status is None:
            assert path.isfile(self._archive_filepath)
            # 1) check its incomplete: has it proper size ?
            size_archive_online = get_remote_file_size(self._archive_url)
            size_archive_local = int(path.getsize(self._archive_filepath))
            if size_archive_online is None:
                logger.critical('impossible to retrieve remote file size.')
                probing_status = 'corrupted'
            elif size_archive_local > size_archive_online:
                logger.critical(f'inconsistent file size, file is bigger than it is supposed to be'
                                f' ({size_archive_online} vs {size_archive_local}).')
                probing_status = 'corrupted'
            elif size_archive_local < size_archive_online:
                logger.debug(f'file_size_online={size_archive_online} != file_size_local={size_archive_local}')
                probing_status = 'incomplete'

        if probing_status is None:
            # 2) size is consistent, check sha256
            assert size_archive_local == size_archive_online
            if not self.is_sha256_consistent():
                probing_status = 'corrupted'

        # archive is there, not installed, with valid sha256 then it must be just downloaded.
        if probing_status is None:
            probing_status = 'downloaded'

        return probing_status

    def __repr__(self) -> str:
        return f'{self.prob_status():10} | {self._name:30} | {self._archive_url}'

    def download(self,
                 force_overwrite: bool = False,
                 nb_attempt: int = 2,
                 previous_status=None) -> str:
        status = previous_status or self.prob_status()
        if force_overwrite and path.isfile(self._archive_filepath):
            logger.debug('remove previously downloaded file and start from scratch.')
            os.remove(self._archive_filepath)
            status = self.prob_status()

        if status == 'downloaded':
            logger.info(f'{path.basename(self._archive_filepath)} is already downloaded.')
            return status

        for attempt in range(nb_attempt):
            if status == 'downloaded':
                logger.debug(f'{path.basename(self._archive_filepath)} is downloaded')
                break
            # check archive file integrity
            if status == 'corrupted':
                logger.warning(f'archive {path.basename(self._archive_filepath)} is corrupted. '
                               f'It will be downloaded again.')
                # if corrupted: remove the archive and start over
                os.remove(self._archive_filepath)
            logger.debug(f'downloading {path.basename(self._archive_filepath)} (attempt {attempt + 1}/{nb_attempt})')
            download_file(self._archive_url, self._archive_filepath)
            status = self.prob_status()
        return status

    def install(self,
                force_overwrite: bool = False,
                no_cleaning: bool = False) -> str:
        """
        Install handles download and untar, and possibly tar remove and launch of an install script.

        """
        # make sure self._status is up to date.
        if force_overwrite:
            self.mark_as_installed(False)

        status = self.prob_status()
        if status == 'installed':
            logger.info(f'{self._install_dir.install_root_path} already exists: skipped')
            return status

        # download (lets try it twice
        status = self.download(previous_status=status)

        if status != 'downloaded':
            # if still not downloaded, it means it failed to download
            logger.critical(f'failed to download {self._name}')
            return status

        # deflating
        logger.debug(f'deflating {path.basename(self._archive_filepath)} to {self._install_dir.install_root_path}')
        untar_file(self._archive_filepath, self._install_dir.install_root_path)

        # optionally: script ?
        if self._install_script_filename is not None:
            logger.debug(f'installation script  {self._install_script_filename}')
            install_script_filepath = path.join(self._install_dir.install_root_path, self._install_script_filename)
            logger.debug(f'applying installation script {install_script_filepath}')
            os.system(install_script_filepath)
        # optionally: clean tar.gz file
        if not no_cleaning:
            logger.debug(f'removing archive file {path.basename(self._archive_filepath)}')
            os.remove(self._archive_filepath)

        # done
        self.mark_as_installed()
        self.upgrade()
        logger.debug(f'done installing {self._name}')
        status = self.prob_status()
        return status

    def upgrade(self):
        # upgrade ?
        status = self.prob_status()
        if status == 'installed':
            # list all sensors.txt files
            file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self._install_dir.install_root_path)
                         for f in filenames if f == 'sensors.txt']
            upgrade_successful = False
            # check version
            for filename in file_list:
                version = get_version_from_csv_file(filename)
                if version is None or version == '1.0':
                    logger.debug(f'upgrading {filename} from version {version} to 1.1')
                    kapture_path = path.dirname(path.dirname(filename))
                    # run inplace upgrade script
                    upgrade_1_0_to_1_1_inplace(kapture_path, None, None, None, 'L2', 'L2')
                    logger.info(f'{kapture_path} - successfully upgraded to 1.1')
                elif version == '1.1':
                    logger.info(f'{filename} already in 1.1, no need to upgrade')
                else:
                    logger.warning(f'{filename} - version {version} is unknown')
            # attempt upgrade of orphan features
            csv_feature_names = ['keypoints.txt', 'descriptors.txt', 'global_features.txt']
            file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self._install_dir.install_root_path)
                         for f in filenames if f in csv_feature_names]
            orphan_lfeat_paths = set()
            orphan_gfeat_paths = set()
            for filename in file_list:
                version = get_version_from_csv_file(filename)
                if version is None or version == '1.0':
                    filename_c = path.abspath(filename).replace('\\', '/').rstrip('/')
                    filename_split = filename_c.split('/')
                    if len(filename) > 1 and \
                            (filename_split[-1] == 'keypoints.txt' or filename_split[-1] == 'descriptors.txt') and \
                            (filename_split[-2] == 'keypoints' or filename_split[-2] == 'descriptors'):
                        orphan_lfeat_paths.add('/'.join(filename_split[0:-2]))
                    elif len(filename) > 1 and \
                            filename_split[-1] == 'global_features.txt' and \
                            filename_split[-2] == 'global_features':
                        orphan_gfeat_paths.add('/'.join(filename_split[0:-2]))
            try:
                if len(orphan_lfeat_paths) > 0 or len(orphan_gfeat_paths) > 0:
                    logger.info(f'upgrade orphan features for {orphan_lfeat_paths} {orphan_gfeat_paths}')
                    upgrade_1_0_to_1_1_orphan_features(list(orphan_lfeat_paths), list(orphan_gfeat_paths))
                upgrade_successful = True
            except Exception as e:
                logger.warning(f'upgrade_1_0_to_1_1_orphan_features {e} for {orphan_lfeat_paths} {orphan_gfeat_paths}')
            return upgrade_successful
        else:
            logger.warning('dataset not yet installed, cannot attempt upgrade')
            return False


def check_licenses(
        install_dir: InstallDir,
        datasets: Dict[str, Dataset]
):
    """
    Check user is aware of licences.
    expected behavior:
    - license == url,
    - only ask user once per url,
    - dont ask user for licenses already approved (and registered in a dedicated file),
    - If user disagree a license, prevent corresponding datasets to be installed.

    :param install_dir:
    :param datasets:
    :return:
    """

    # regroup licenses
    license_datasets = {}  # license_url -> [list of datasets referring to that license]
    for name, dataset in datasets.items():
        if dataset.license_url is not None:
            license_datasets.setdefault(dataset.license_url, list())
            license_datasets[dataset.license_url].append(name)
    logger.debug(f'dataset refer {len(license_datasets)} licences.')

    # retrieve already agreed licenses.
    licenses_already_agreed = install_dir.read_licenses()
    # and filter only the ones we are concerned about.
    licenses_already_agreed = {
        license_url
        for license_url in licenses_already_agreed
        if license_url in license_datasets
    }

    # removes from license_datasets, the the ones already agreed upon.
    for license_to_skip in licenses_already_agreed:
        del license_datasets[license_to_skip]

    if not license_datasets:
        # no license to agree, job done
        return

    # some licenses remains to be approved
    logger.info(f'{len(license_datasets)} to be approved:')
    licenses_already_agreed = set()
    # prompt user for each one
    for license_url, dataset_names in license_datasets.items():
        r = requests.get(license_url, allow_redirects=True)
        if r.status_code != requests.codes.ok:
            raise ConnectionError(f'unable to grab {license_url} (code:{r.status_code})')
        logger.info(f'here after the license terms for : {", ".join(dataset_names)}')
        wrapper = textwrap.TextWrapper(width=80)
        print(wrapper.fill(text=r.text))
        print('----------------------')
        print(f'from: {license_url}')
        agree = ask_confirmation('do you agree on terms ?')
        if not agree:
            # in case of disagreement: remove all dataset with that license,
            licenses_already_agreed.add(license_url)
            for name_of_disagreement in dataset_names:
                logger.critical(f'removing dataset {name_of_disagreement} due to license disagreement.')
                del datasets[name_of_disagreement]

    logger.debug('update list of agreed licences.')
    for license_to_skip in licenses_already_agreed:
        del license_datasets[license_to_skip]
    install_dir.update_licenses(license_datasets)


def kapture_download_dataset(args, index_filepath: str):
    """
    Do the dataset download command
    """
    global_status = 0
    install_dir = InstallDir(index_filepath=index_filepath, install_dir_path=args.install_path)
    if args.cmd == 'update':
        logger.info(f'updating dataset list from {args.repo} ...')
        index_remote_url = args.repo + '/' + INDEX_FILENAME
        logger.debug(f'retrieving index at {index_remote_url}')
        r = requests.get(index_remote_url, allow_redirects=True)
        if r.status_code != requests.codes.ok:
            raise ConnectionError(f'unable to grab {index_remote_url} (code:{r.status_code})')
        with open(index_filepath, 'wt') as f:
            f.write(r.text)
        datasets = install_dir.load_datasets_from_file()
        logger.info(f'dataset index retrieved successfully: {len(datasets)} datasets')

    elif args.cmd == 'list':
        logger.info(f'listing dataset {index_filepath} ...')
        datasets = install_dir.load_datasets_from_file(filter_patterns=args.dataset)
        for name, dataset in datasets.items():
            status = dataset.prob_status(check_online=args.full)
            if status == "not reachable" or status == "incomplete" or status == "corrupted":
                global_status = 1
            print(f'{status:^16}| {name:40} | {dataset.url}')

    elif args.cmd == 'install':
        logger.debug(f'will install dataset: {args.dataset} ...')
        datasets = install_dir.load_datasets_from_file(filter_patterns=args.dataset)
        if len(datasets) == 0:
            raise ValueError('There is no matching dataset.'
                             ' Make sure you used quotes (") to prevent shell interpreting * wildcard.')

        # ask for license agreement
        check_licenses(install_dir=install_dir, datasets=datasets)

        logger.info(f'{len(datasets)} dataset will be installed.')
        for name, dataset in datasets.items():
            logger.info(f'{name}: starting installation  ...')
            status = dataset.install(force_overwrite=args.force, no_cleaning=args.no_cleaning)
            logger.info(f'{name} install: ' + 'successful' if status == 'installed' else 'failed')

    elif args.cmd == 'upgrade':
        logger.debug(f'will install dataset: {args.dataset} ...')
        datasets = install_dir.load_datasets_from_file(filter_patterns=args.dataset)
        if len(datasets) == 0:
            raise ValueError('There is no matching dataset.'
                             ' Make sure you used quotes (") to prevent shell interpreting * wildcard.')

        logger.info(f'{len(datasets)} dataset will be installed.')
        for name, dataset in datasets.items():
            logger.info(f'{name}: starting installation  ...')
            dataset.upgrade()

    elif args.cmd == 'download':
        logger.debug(f'will download dataset: {args.dataset} ...')
        datasets = install_dir.load_datasets_from_file(filter_patterns=args.dataset)
        if len(datasets) == 0:
            raise ValueError('There is no matching dataset.'
                             ' Make sure you used quotes (") to prevent shell interpreting * wildcard.')
        logger.info(f'{len(datasets)} dataset will be downloaded.')
        for name, dataset in datasets.items():
            logger.info(f'downloading {name} ...')
            dataset.download(force_overwrite=args.force)

    return global_status


def kapture_download_dataset_cli():
    """
    Parse the kapture_download_dataset command line .
    """
    parser = argparse.ArgumentParser(description='download kapture datasets.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.INFO, const=logging.INFO,
        action=kapture.utils.logging.VerbosityParser,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('--install_path', default=path.normpath(path.join(DEFAULT_DATASET_PATH)),
                        help=f'path to index files listing all datasets'
                             f' [{path.normpath(path.join(DEFAULT_DATASET_PATH))}]')
    subparsers = parser.add_subparsers(help='sub-command help', dest='cmd')
    ####################################################################################################################
    # create the parser for the "update" command
    parser_update = subparsers.add_parser('update', help='update dataset index')
    parser_update.set_defaults(cmd='update')
    parser_update.add_argument('--repo', default=DEFAULT_REPOSITORY_URL,
                               help='url of the repository.')
    ####################################################################################################################
    parser_list = subparsers.add_parser('list', help='display dataset index')
    parser_list.set_defaults(cmd='list')
    parser_list.add_argument('dataset', nargs='*', default=[])
    parser_list.add_argument('--full', action='store_true', default=False,
                             help='Display status and size of remote dataset.')
    ####################################################################################################################
    parser_install = subparsers.add_parser('install', help='install dataset')
    parser_install.set_defaults(cmd='install')
    parser_install.add_argument('-f', '--force', action='store_true', default=False,
                                help='Force installation even if dataset has already been installed.')
    parser_install.add_argument('--no_cleaning', action='store_true', default=False,
                                help='Do not delete downloaded tar.gz file.')
    parser_install.add_argument('dataset', nargs='*', default=[],
                                help='name of the dataset to download. Can use unix-like wildcard.')
    ####################################################################################################################
    parser_download = subparsers.add_parser('download', help='download dataset, without installing it')
    parser_download.set_defaults(cmd='download')
    parser_download.add_argument('-f', '--force', action='store_true', default=False,
                                 help='Force installation even if dataset has already been installed.')
    parser_download.add_argument('dataset', nargs='*', default=[],
                                 help='name of the dataset to download. Can use unix-like wildcard.')
    ####################################################################################################################
    parser_install = subparsers.add_parser('upgrade', help='upgrade dataset to 1.1')
    parser_install.set_defaults(cmd='upgrade')
    parser_install.add_argument('dataset', nargs='*', default=[],
                                help='name of the dataset to download. Can use unix-like wildcard.')
    ####################################################################################################################

    args = parser.parse_args()

    logger.setLevel(args.verbose or logging.INFO)
    logger.debug(f'{sys.argv[0]} \\\n' + '  \\\n'.join(
        '--{:20} {:100}'.format(k, str(v))
        for k, v in vars(args).items()))

    try:
        index_filepath = path.join(args.install_path, INDEX_FILENAME)
        if not args.cmd:
            parser.print_help()
            logger.critical(f'Choose command among [ {" | ".join(subparsers.choices)} ]')
            exit(-1)
        return kapture_download_dataset(args, index_filepath)

    except Exception as e:
        logger.critical(e)
        if logger.getEffectiveLevel() >= logging.DEBUG:
            raise e
        return -1


if __name__ == '__main__':
    sys.exit(kapture_download_dataset_cli())
