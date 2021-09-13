# -*- coding: utf-8 -*-
"""Tests for the similarity measure.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest
from beartype.roar import BeartypeException

import cfs

# Current directory
HERE = os.path.dirname(__file__)


def X1_file():
    """Define coordinate file."""
    return os.path.join(HERE, 'X1.dat')


def X1():
    """Correlated coordinates."""
    x = np.linspace(0, np.pi, 1000)
    return np.array([
        np.cos(x), np.cos(x + np.pi / 6),
    ]).T


def X1_result(str):
    """Correlated coordinates results."""
    return {
        'correlation': 0.9697832,
        'GY': 0.99960494,
        'JSD': 0.93378449,
        'NMI_joint': 0.70474596,
        'NMI_max': 0.81167031,
        'NMI_arithmetic': 0.82680467,
        'NMI_geometric': 0.82694844,
        'NMI_min': 0.84251415,
    }[str]


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
            cfs.similarity._entropy(p), result,
        )
    else:
        with pytest.raises(error):
            cfs.similarity._entropy(p)


@pytest.mark.parametrize('p, q, result, error', [
    ([1., 1.], [1., 1.], 0, None),
    ([1., 1.], [1 / np.e, 1.], 1, None),
    ([1., 1., 1.], [1., 1.], None, ValueError),
    ([1, 1], [1, 1], 0, BeartypeException),
])
def test__kullback(p, q, result, error):
    if not error:
        np.testing.assert_almost_equal(
            cfs.similarity._kullback(p, q), result,
        )
    else:
        with pytest.raises(error):
            cfs.similarity._kullback(p, q)


@pytest.mark.parametrize('X, error', [
    (np.random.uniform(size=(100, 20)), None),
    (np.random.uniform(size=(10, 5)), None),
    (np.random.uniform(size=10), BeartypeException),
    (np.random.uniform(size=(10, 5, 5)), BeartypeException),
])
def test__standard_scaler(X, error):
    if not error:
        Xscaled = cfs.similarity._standard_scaler(X)
        np.testing.assert_array_almost_equal(
            np.mean(Xscaled), np.zeros_like(Xscaled),
        )
        np.testing.assert_array_almost_equal(
            np.std(Xscaled), np.ones_like(Xscaled),
        )
    else:
        with pytest.raises(error):
            cfs.similarity._standard_scaler(X)


@pytest.mark.parametrize('x, y, kwargs, error', [
    (np.random.uniform(size=10000), np.random.uniform(size=10000), {}, None),
    (
        np.random.uniform(size=10000),
        np.random.uniform(size=10000),
        {'bins': 10},
        None
    ),
])
def test__estimate_densities(x, y, kwargs, error):
    if not error:
        densities = cfs.similarity._estimate_densities(x, y, **kwargs)
        _, _, px, py = densities

        # check normalization
        for p in densities:
            np.testing.assert_almost_equal(np.sum(p), 1)
    else:
        with pytest.raises(error):
            cfs.similarity._estimate_densities(x, y, **kwargs)


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
            cfs.similarity._correlation(X), result,
        )
    else:
        with pytest.raises(error):
            cfs.similarity._correlation(X)


@pytest.mark.parametrize('metric, kwargs, X, result, error', [
    ('correlation', {}, X1(), X1_result('correlation'), None),
    (
        'correlation',
        {'online': True},
        X1_file(),
        X1_result('correlation'),
        None,
    ),
    ('GY', {}, X1(), X1_result('GY'), None),
    ('JSD', {}, X1(), X1_result('JSD'), None),
    ('NMI', {'normalize_method': 'joint'}, X1(), X1_result('NMI_joint'), None),
    ('NMI', {'normalize_method': 'max'}, X1(), X1_result('NMI_max'), None),
    (
        'NMI',
        {'normalize_method': 'arithmetic'},
        X1(),
        X1_result('NMI_arithmetic'),
        None,
    ),
    (
        'NMI',
        {'normalize_method': 'geometric'},
        X1(),
        X1_result('NMI_geometric'),
        None,
    ),
    ('NMI', {'normalize_method': 'min'}, X1(), X1_result('NMI_min'), None),
])
def test_Similarity(metric, kwargs, X, result, error):
    if not error:
        sim = cfs.Similarity(metric=metric, **kwargs)
        sim.fit(X)
        np.testing.assert_almost_equal(
            sim.matrix_[-1, 0], result,
        )
    else:
        with pytest.raises(error):
            sim = cfs.Similarity(metric=metric, **kwargs)
            sim.fit(X)
