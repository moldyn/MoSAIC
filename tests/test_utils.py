# -*- coding: utf-8 -*-
"""Tests for the utils module.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import os.path

import numpy as np
import pytest

import mosaic

# Current directory
HERE = os.path.dirname(__file__)
TEST_FILE_DIR = os.path.join(HERE, 'test_files')


def Xrand(N):
    """Create random symmetric NxN matrix M with diag(M) = 1."""
    mat = np.random.uniform(size=N**2).reshape(N, N)
    mat[np.diag_indices_from(mat)] = 1
    return 0.5 * (mat + mat.T)


def clust_ref(idx):
    """Get filename and expected clusters, idx=1, 2, 3."""
    filename = os.path.join(TEST_FILE_DIR, f'clust{idx}.dat')
    clusters_list = {
        1: [[0, 1, 2], [4, 5], [3], [6, 7, 8, 9]],
        2: [[0, 1], [4, 5], [2, 3], [6, 7], [8, 9]],
        3: [[0, 1, 2, 3, 4, 5, 6, 8, 9, 7]],
    }[idx]
    clusters = np.empty(len(clusters_list), dtype=object)
    clusters[:] = clusters_list
    return filename, clusters


@pytest.mark.parametrize(
    'filename, clusters_ref', [clust_ref(idx) for idx in (1, 2, 3)],
)
def test_load_clusters(filename, clusters_ref):
    """Test loading clusters"""
    # load clusters
    clusters = mosaic.utils.load_clusters(filename)

    assert len(clusters) == len(clusters_ref)
    for idx, cluster in enumerate(clusters):
        np.testing.assert_almost_equal(
            clusters_ref[idx], cluster,
        )


@pytest.mark.parametrize(
    'clusters_ref', [clust_ref(idx)[1] for idx in (1, 2, 3)],
)
def test_save_clusters(clusters_ref, tmpdir):
    """Test loading clusters"""
    # save and load clusters
    filename = str(tmpdir.mkdir('sub').join('load_clusters_test'))
    mosaic.utils.save_clusters(filename, clusters_ref)
    clusters = mosaic.utils.load_clusters(filename)

    assert len(clusters) == len(clusters_ref)
    for idx, cluster in enumerate(clusters):
        np.testing.assert_almost_equal(
            clusters_ref[idx], cluster,
        )


@pytest.mark.parametrize(
    'X', [Xrand(N=N) for N in (10, 30, 50, 100)],
)
def test_save_load_clusters(X, tmpdir):
    """Test save/load clusters against random matrices."""
    clust = mosaic.Clustering(mode='CPM')
    clust.fit(X)

    # save and load clusters
    filename = str(tmpdir.mkdir('sub').join('load_clusters_test'))
    mosaic.utils.save_clusters(filename, clust.clusters_)
    clusters = mosaic.utils.load_clusters(filename)

    assert len(clusters) == len(clust.clusters_)
    for i, cluster in enumerate(clusters):
        np.testing.assert_almost_equal(
            clust.clusters_[i], cluster,
        )
