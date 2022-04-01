# -*- coding: utf-8 -*-
import pathlib
import sys
from collections import defaultdict

import setuptools

# check for python version
if sys.version_info < (3, 8):
    raise SystemExit('Python 3.8+ is required!')


def get_extra_requirements(path, add_all=True):
    """Parse extra-requirements file."""

    with open(path) as depfile:
        extra_deps = defaultdict(set)
        for line in depfile:
            if not line.startswith('#'):
                if ':' not in line:
                    raise ValueError(
                        f'Dependency in {path} not correct formatted: {line}',
                    )
                dep, tags = line.split(':')
                tags = {tag.strip() for tag in tags.split(',')}
                for tag in tags:
                    extra_deps[tag].add(dep)

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = {
                tag for tags in extra_deps.values() for tag in tags
            }

    return extra_deps


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / 'README.md').read_text()

# This call to setup() does all the work
setuptools.setup(
    name='mosaic-clustering',
    version='0.2.0',
    description='Correlation based feature selection for MD data',
    long_description=README,
    long_description_content_type='text/markdown',
    keywords=[
        'feature selection',
        'MD analysis',
    ],
    author='braniii',
    url='https://github.com/moldyn/feature_selection',
    license='BSD 3-Clause License',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'mosaic = mosaic.__main__:main',
        ],
    },
    install_requires=[
        'numpy>=1.21.0',
        'pandas',
        'beartype>=0.10.4',
        'scipy',
        'scikit-learn',
        'igraph',
        'leidenalg>=0.8.0',
        'umap-learn',
        'click>=7.0.0',
        'typing_extensions>=3.9.0;python_version<"3.9"',
        'prettypyplot',
    ],
    extras_require=get_extra_requirements('extra-requirements.txt')
)
