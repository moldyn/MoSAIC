# -*- coding: utf-8 -*-
"""Tests for the correlation measures.

MIT License
Copyright (c) 2022, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest
from beartype.roar import BeartypeException
from sklearn.preprocessing import StandardScaler

import mosaic

# Current directory
HERE = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(HERE, 'test_files')


def X1_file(only_first_line=False):
    """Define coordinate file."""
    if only_first_line:
        return os.path.join(TEST_FILE_DIR, 'X1_single_row.dat')
    return os.path.join(TEST_FILE_DIR, 'X1.dat')


def no_file():
    """Not existing file."""
    return os.path.join(TEST_FILE_DIR, 'no_file')


@pytest.mark.parametrize('filename, result', [
    (
        X1_file(),
        (
            np.array([[1, 0.9697832], [0.9697832, 1]]),
            1000,
            2,
        ),
    ),
    (
        X1_file(only_first_line=True),
        (np.full((2, 2), np.nan), 1, 2),
    ),
])
def test__welford_correlation(filename, result):
    wc_res = mosaic._correlation_utils._welford_correlation(
        filename=filename, dtype=np.float64,
    )

    for idx in range(3):
        np.testing.assert_almost_equal(wc_res[idx], result[idx])


@pytest.mark.parametrize('p, result, error', [
    ([1.0, np.e, 1.0], -np.e, None),
    (np.ones(10), 0, None),
    (np.arange(10).astype(np.float64), -79.05697962199447, None),
    (np.arange(10).astype(np.float16), -79.0625, None),
    (np.arange(10).astype(int), None, BeartypeException),
    ([1, 2, 1], None, BeartypeException),
    (['a', 'b'], None, BeartypeException),
    (15, None, BeartypeException),
])
def test__entropy(p, result, error):
    if not error:
        np.testing.assert_almost_equal(
            mosaic._correlation_utils._entropy(p), result,
        )
    else:
        with pytest.raises(error):
            mosaic._correlation_utils._entropy(p)


@pytest.mark.parametrize('p, q, result, error', [
    ([1., 1.], [1., 1.], 0, None),
    ([1., 1.], [1 / np.e, 1.], 1, None),
    ([1., 1., 1.], [1., 1.], None, ValueError),
    ([1, 1], [1, 1], 0, BeartypeException),
])
def test__kullback(p, q, result, error):
    if not error:
        np.testing.assert_almost_equal(
            mosaic._correlation_utils._kullback(p, q), result,
        )
    else:
        with pytest.raises(error):
            mosaic._correlation_utils._kullback(p, q)


@pytest.mark.parametrize('x, y, kwargs, error', [
    (np.random.uniform(size=10000), np.random.uniform(size=10000), {}, None),
    (
        np.random.uniform(size=10000),
        np.random.uniform(size=10000),
        {'bins': 10},
        None,
    ),
])
def test__estimate_densities(x, y, kwargs, error):
    if not error:
        densities = mosaic._correlation_utils._estimate_densities(
            x, y, **kwargs,
        )
        _, _, px, py = densities

        # check normalization
        for p in densities:
            np.testing.assert_almost_equal(np.sum(p), 1)
    else:
        with pytest.raises(error):
            mosaic._correlation_utils._estimate_densities(x, y, **kwargs)


@pytest.mark.parametrize('X, result, error', [
    (
        np.sqrt(3 / 2) * np.array([[-1, 1], [0, 0], [1, -1]]),
        np.array([[1, -1], [-1, 1]]),
        None,
    ),
    (np.arange(10), None, BeartypeException),
    (np.arange(10).astype(np.float64), None, BeartypeException),
])
def test__correlation(X, result, error):
    if not error:
        np.testing.assert_array_almost_equal(
            mosaic._correlation_utils._correlation(X), result,
        )
    else:
        with pytest.raises(error):
            mosaic._correlation_utils._correlation(X)


@pytest.mark.parametrize('n_samples, n_features, dtype', [
    (100000, 10, np.float16),
    (100000, 10, np.float32),
    (100000, 10, np.float64),
    (10000, 100, np.float16),
    (10000, 100, np.float32),
    (1000, 1000, np.float16),
    (1000, 1000, np.float32),
])
def test__correlation_dtype(n_samples, n_features, dtype):
    np.random.seed(42)

    # create random data
    X = StandardScaler().fit_transform(
        np.random.normal(
            size=(n_samples, n_features),
        ).astype(dtype),
    )

    # check if results is symmetric and diagonal close to 1
    mosaic._correlation_utils._correlation(X)
