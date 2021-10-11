# -*- coding: utf-8 -*-
"""Tests for the clustering module.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import mosaic


def X1():
    return np.array([[1.0, 0.1, 0.9], [0.1, 1.0, 0.0], [0.8, 0.1, 1.0]])


def X1_sorted():
    return np.array([[1.0, 0.9, 0.1], [0.8, 1.0, 0.1], [0.1, 0.0, 1.0]])


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
            mosaic.clustering._coarse_clustermatrix(cluster_list, mat),


@pytest.mark.parametrize('mode, kwargs, X, Xresult, n_clusters, error', [
    ('modularity', {'n_neighbors': 1}, X1(), X1_sorted(), 2, None),
    ('modularity', {'n_neighbors': 2}, X1(), X1_sorted(), 1, None),
    ('modularity', {'n_neighbors': 3}, X1(), None, None, ValueError),
    ('CPM', {}, X1(), X1_sorted(), 2, None),
    ('CPM', {'resolution_parameter': 0.9}, X1(), X1_sorted(), 3, None),
    ('CPM', {'resolution_parameter': 0.05}, X1(), X1_sorted(), 1, None),
])
def test_Similarity(mode, kwargs, X, Xresult, n_clusters, error):
    if not error:
        clust = mosaic.Clustering(mode=mode, **kwargs)
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
