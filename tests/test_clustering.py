# -*- coding: utf-8 -*-
"""Tests for the clustering module.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import mosaic


@pytest.fixture
def X():
    return np.array([[1.0, 0.1, 0.9], [0.1, 1.0, 0.1], [0.9, 0.1, 1.0]])


@pytest.fixture
def Xresult():
    return np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])


@pytest.mark.parametrize('clusters, mat, result, error', [
    (
        [[0], [1, 2]],
        np.array([[1.0, 0.8, 0.6], [0.8, 0.6, 0.4], [0.6, 0.4, 0.2]]),
        np.array([[1.0, 0.7], [0.7, 0.4]]),
        None,
    ),
    (
        [[0], [1], [2]],
        np.array([[1.0, 0.8, 0.6], [0.8, 0.6, 0.4], [0.6, 0.4, 0.2]]),
        np.array([[1.0, 0.8, 0.6], [0.8, 0.6, 0.4], [0.6, 0.4, 0.2]]),
        None,
    ),
    (
        [[0, 1, 2]],
        np.array([[1.0, 0.8, 0.6], [0.8, 0.6, 0.4], [0.6, 0.4, 0.2]]),
        np.array([[0.6]]),
        None,
    ),
])
def test__coarse_clustermatrix(clusters, mat, result, error):
    cluster_list = np.empty(len(clusters), dtype=object)
    cluster_list[:] = clusters
    if not error:
        np.testing.assert_array_almost_equal(
            mosaic.clustering._coarse_clustermatrix(cluster_list, mat), result,
        )
    else:
        with pytest.raises(error):
            mosaic.clustering._coarse_clustermatrix(cluster_list, mat)


@pytest.mark.parametrize('mode, kwargs, n_clusters, error', [
    ('modularity', {'n_neighbors': 1}, 2, None),
    ('modularity', {'n_neighbors': 2}, 1, None),
    ('modularity', {'n_neighbors': 3}, None, ValueError),
    ('CPM', {}, 2, None),
    ('CPM', {'resolution_parameter': 0.9}, 3, None),
    ('CPM', {'resolution_parameter': 0.05}, 1, None),
    ('linkage', {'resolution_parameter': 0.9}, 3, None),
    ('linkage', {'resolution_parameter': 0.05}, 1, None),
    (
        'linkage',
        {'resolution_parameter': 0.05, 'n_neighbors': 3},
        None,
        NotImplementedError,
    ),
    ('CPM', {'n_clusters': 1}, None, NotImplementedError),
    ('linkage', {'n_clusters': 1}, None, NotImplementedError),
    ('modularity', {'n_clusters': 1}, None, NotImplementedError),
    ('kmedoids', {}, None, TypeError),
    ('kmedoids', {'n_clusters': 2}, 2, None),
    (
        'kmedoids',
        {'n_neighbors': 2, 'n_clusters': 2},
        None,
        NotImplementedError,
    ),
    ('linkage', {'n_neighbors': 2}, None, NotImplementedError),
    ('CPM', {'weighted': False}, None, NotImplementedError),
    ('linkage', {'weighted': False}, None, NotImplementedError),
    (
        'kmedoids',
        {'n_neighbors': 2, 'resolution_parameter': 0.9},
        None,
        NotImplementedError,
    ),
    ('modularity', {'resolution_parameter': 0.9}, None, NotImplementedError),
])
def test_Clustering(mode, kwargs, n_clusters, error, X, Xresult):
    if not error:
        clust = mosaic.Clustering(mode=mode, **kwargs)
        clust.fit(X)
        np.testing.assert_array_almost_equal(
            clust.matrix_, Xresult,
        )
        np.testing.assert_almost_equal(
            len(clust.clusters_), n_clusters,
        )
        # refit to test resetting
        clust.fit(X)
        np.testing.assert_array_almost_equal(
            clust.matrix_, Xresult,
        )
        np.testing.assert_almost_equal(
            len(clust.clusters_), n_clusters,
        )
    else:
        with pytest.raises(error):
            clust = mosaic.Clustering(mode=mode, **kwargs)
            clust.fit(X)
