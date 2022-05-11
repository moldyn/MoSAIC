# -*- coding: utf-8 -*-
"""Tests for the similarity measure.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest
from beartype.roar import BeartypeException

import mosaic

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


def X1_result(mode):
    """Correlated coordinates results."""
    return {
        'correlation': 0.9697832,
        'GY': 0.94966701,
        'JSD': 0.67786610,
        'NMI_joint': 0.54114068,
        'NMI_max': 0.68108618,
        'NMI_arithmetic': 0.70225994,
        'NMI_geometric': 0.702599541,
        'NMI_min': 0.72479244,
    }[mode]


@pytest.mark.parametrize('X, error', [
    (np.random.uniform(size=(100, 20)), None),
    (np.random.uniform(size=(10, 5)), None),
    (np.random.uniform(size=10), BeartypeException),
    (np.random.uniform(size=(10, 5, 5)), BeartypeException),
])
def test__standard_scaler(X, error):
    if not error:
        Xscaled = mosaic.similarity._standard_scaler(X)
        np.testing.assert_array_almost_equal(
            np.mean(Xscaled), np.zeros_like(Xscaled),
        )
        np.testing.assert_array_almost_equal(
            np.std(Xscaled), np.ones_like(Xscaled),
        )
    else:
        with pytest.raises(error):
            mosaic.similarity._standard_scaler(X)


@pytest.mark.parametrize('metric, kwargs, X, result, error', [
    ('correlation', {}, X1(), X1_result('correlation'), None),
    (
        'correlation',
        {'low_memory': True},
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
        sim = mosaic.Similarity(metric=metric, **kwargs)
        sim.fit(X)
        np.testing.assert_almost_equal(
            sim.matrix_[-1, 0], result,
        )
    else:
        with pytest.raises(error):
            sim = mosaic.Similarity(metric=metric, **kwargs)
            sim.fit(X)
