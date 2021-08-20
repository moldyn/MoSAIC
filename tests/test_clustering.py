# -*- coding: utf-8 -*-
"""Tests for the clustering module.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import numpy as np
import pytest

import cfs


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
            cfs.clustering._coarse_clustermatrix(cluster_list, mat), result,
        )
    else:
        with pytest.raises(error):
            cfs.clustering._coarse_clustermatrix(cluster_list, mat),
