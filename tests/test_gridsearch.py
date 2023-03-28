# -*- coding: utf-8 -*-
"""Tests for the gridsearch.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import mosaic


def X():
    """Correlated coordinates."""
    # fix random seed
    np.random.seed(42)

    x = np.linspace(0, 2 * np.pi, 1000)
    rand_offsets = np.random.uniform(
        low=-np.pi / 6, high=np.pi / 6, size=10,
    )

    traj = np.array([
        *[np.sin(x + xi) for xi in rand_offsets],
        *[np.cos(x + xi) for xi in rand_offsets],
        *[np.zeros_like(x) for _ in rand_offsets],
    ]).T
    return traj + np.random.normal(size=traj.shape, scale=.2)


@pytest.mark.parametrize('params, clust_kwargs, X, best_index, error', [
    ({'resolution_parameter': np.linspace(0, 1, 7)}, {}, X(), 1, None),
    (
        {'n_clusters': np.arange(3, 30, 3)},
        {'mode': 'kmedoids', 'n_clusters': 2},
        X(),
        3,
        None,
    ),
    ({}, {}, X(), None, ValueError),
])
def test_GridSearchCV(params, clust_kwargs, X, best_index, error):
    if not error:
        print(mosaic.Similarity().fit_transform(X))
        search = mosaic.GridSearchCV(
            similarity=mosaic.Similarity(),
            clustering=mosaic.Clustering(**clust_kwargs),
            param_grid=params,
        )
        search.fit(X)
        assert search.best_index_ == best_index
    else:
        with pytest.raises(error):
            search = mosaic.GridSearchCV(
                similarity=mosaic.Similarity(),
                clustering=mosaic.Clustering(**clust_kwargs),
                param_grid=params,
            )
            search.fit(X)
