# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import setuptools
import warnings
import os.path as path
from glob import glob
from subprocess import SubprocessError, check_call
import os
import tempfile
import sys

HERE = path.normpath(path.dirname(__file__))


# Documentation ########################################################################################################
# setuptools only support md documentation: convert from asciidoc
def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def read_doc(filepath):
    try:  # from asciidoc to markdown
        # https://tinyapps.org/blog/201701240700_convert_asciidoc_to_markdown.html
        with tempfile.TemporaryDirectory() as tmpdirname:
            xml_filepath = path.join(tmpdirname, 'README.xml')
            md_filepath = path.join(tmpdirname, 'README.md')
            use_shell = sys.platform.startswith("win")
            check_call(['asciidoctor', '-b', 'docbook', filepath, '-o', xml_filepath], shell=use_shell)
            check_call(['pandoc', '-f', 'docbook', '-t', 'markdown_strict', xml_filepath, '-o', md_filepath],
                       shell=use_shell)
            content = read_file(md_filepath)

    except (FileNotFoundError, SubprocessError):
        warnings.warn('cannot convert asciidoc to markdown.')
        content = read_file(filepath)

    return content


readme_filepath = path.join(HERE, 'README.adoc')
long_description = read_doc(readme_filepath)

########################################################################################################################
setuptools.setup(
    # description
    name='kapture',
    version="1.1.7",
    author="naverlabs",
    author_email="kapture@naverlabs.com",
    description="kapture: file format for SfM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naver/kapture/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],

    # dependencies
    python_requires='>=3.6',
    install_requires=[
        'dataclasses>=0.3;python_version<\'3.7\'',
        'numpy>=1.16',
        'numpy-quaternion>=2019.3.18.14.33.20',
        'numba>=0.42',
        'matplotlib>=3.0',
        'scipy>=1.4',
        'tqdm>=4.30',
        'Pillow>=8.1.1',
        'piexif==1.1.3',
        'requests>=2.21',
        'pyyaml>=5.1',
        'wcmatch>=5.0',
        'tabulate>=0.8.7',
        'pytz>=2021.1'
    ],
    extras_require={
        'dev': ['pytest'],
    },
    # sources
    packages=setuptools.find_packages(),
    scripts=[f for f in glob(os.path.join(HERE, 'tools', '*.py'))],
)
