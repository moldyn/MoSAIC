# -*- coding: utf-8 -*-
"""Tests for the tools module.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import mosaic


def Xrand(N):
    mat = np.random.uniform(size=N**2).reshape(N, N)
    mat[np.diag_indices_from(mat)] = 1
    mat = 0.5 * (mat + mat.T)
    return mat


@pytest.mark.parametrize('X', [Xrand(N=10), Xrand(N=30), Xrand(N=50)])
def test_load_clusters(X, tmpdir):
    clust = mosaic.Clustering(mode='CPM')
    clust.fit(X)

    # save clusters
    filename = str(tmpdir.mkdir('sub').join('load_clusters_test'))
    clusters_string = np.array(
        [
            ' '.join([str(state) for state in cluster])
            for cluster in clust.clusters_
        ],
        dtype=str,
    )
    np.savetxt(
        filename,
        clusters_string,
        fmt='%s',
    )

    clusters = mosaic.load_clusters(filename)
    assert len(clusters) == len(clust.clusters_)
    for i in range(len(clusters)):
        np.testing.assert_almost_equal(
            clust.clusters_[i], clusters[i],
        )
