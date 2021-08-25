# -*- coding: utf-8 -*-
import pathlib
import sys

import setuptools

# check for python version
if sys.version_info < (3, 8):
    raise SystemExit('Python 3.8+ is required!')

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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas',
        'beartype>=0.8.1',
        'scipy',
        'scikit-learn',
        'leidenalg>=0.8.0',
        'umap-learn',
        'click>=7.0.0',
    ],
)
