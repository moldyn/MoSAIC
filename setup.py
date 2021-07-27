# -*- coding: utf-8 -*-
import pathlib
import sys

import setuptools

# check for python version
if sys.version_info < (3, 6):
    raise SystemExit('Python 3.6+ is required!')

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / 'README.md').read_text()

# This call to setup() does all the work
setuptools.setup(
    name='cfs',
    version='0.1.0',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'leidenalg',
        'umap-learn',
        'numba',
        'decorit',
    ],
)
