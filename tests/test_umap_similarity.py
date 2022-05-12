# -*- coding: utf-8 -*-
"""Tests for the umap similarity.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""

import numpy as np
import pytest

import mosaic


def X(idx):
    """Correlation matrices."""
    if idx == 0:
        return np.array([
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.6],
            [0.0, 0.1, 0.6, 1.0],
        ])
    if idx == 1:
        return np.array([
            [1.0, 0.9, 0.1, 0.0, 0.0],
            [0.9, 1.0, 0.2, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.6, 0.7],
            [0.0, 0.2, 0.6, 1.0, 0.8],
            [0.0, 0.1, 0.7, 0.8, 1.0],
        ])
    raise ValueError(f'idx={idx} is not defined')


def X_result(idx, n=None):
    if idx == 0:
        if n in [None, 2]:
            return np.array([
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ])
        elif n == 4:
            return np.ones((4, 4))
        raise ValueError(
            f'idx={idx} with {n} neighbors is not defined',
        )
    if idx == 1:
        if n in [None, 2, 3]:
            return np.array([
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ])
        raise ValueError(
            f'idx={idx} with {n} neighbors is not defined',
        )


@pytest.mark.parametrize('kwargs, X, result, error', [
    ({}, X(idx=0), X_result(idx=0), None),
    ({'n_neighbors': 2}, X(idx=0), X_result(idx=0, n=2), None),
    ({}, X(idx=1), X_result(idx=1), None),
    ({'n_neighbors': 2}, X(idx=1), X_result(idx=1, n=2), None),
    ({'n_neighbors': 3}, X(idx=1), X_result(idx=1, n=3), None),
    ({'n_neighbors': 4}, X(idx=0), X_result(idx=0, n=4), ValueError),
    ({'n_neighbors': 10}, X(idx=0), None, ValueError),
])
def test_UMAPSimilarity(kwargs, X, result, error):
    if not error:
        umap = mosaic.UMAPSimilarity(**kwargs)
        umap.fit(X)
        np.testing.assert_allclose(
            umap.matrix_, result, atol=0.3,
        )
    else:
        with pytest.raises(error):
            umap = mosaic.UMAPSimilarity(**kwargs)
            umap.fit(X)
