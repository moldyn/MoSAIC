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

import mosaic

# Current directory
HERE = os.path.dirname(__file__)


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
        np.array([[1.0, 1.0], [0.9, 0.8], [0.8, 0.6]]),
        np.array([[49, 44], [44, 40]]) / 60,
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
