# -*- coding: utf-8 -*-
"""Class for clustering the correlation matrices.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
__all__ = ['Clustering']  # noqa: WPS410

from typing import Any, Dict, Optional

import igraph as ig
import leidenalg as la
import numpy as np
from beartype import beartype
from sklearn.neighbors import NearestNeighbors

from cfs.typing import (
    ClusteringModeString,
    Float2DArray,
    FloatMatrix,
    Index1DArray,
    NumInRange0to1,
    Object1DArray,
    PositiveInt,
)


@beartype
def _coarse_clustermatrix(
    clusters: Object1DArray, mat: FloatMatrix,
) -> FloatMatrix:
    """Construct a coarse cluster matrix by averaging over all clusters."""
    if len(clusters) == len(mat):
        return np.copy(mat)

    nclusters = len(clusters)
    return np.array([
        mat[np.ix_(clusters[idx_i], clusters[idx_j])].mean()
        for idx_i, idx_j in np.ndindex(nclusters, nclusters)
    ]).reshape(nclusters, nclusters)


@beartype
def _sort_coarse_clustermatrix(coarsemat: FloatMatrix) -> Index1DArray:
    """Return indices which sort clusters to minimize off-diagonal values."""
    nclusters = len(coarsemat)
    clusters = np.empty(nclusters, dtype=object)
    clusters[:] = [[i] for i in range(nclusters)]  # noqa: WPS362

    for _ in range(nclusters - 1):
        cmat = _coarse_clustermatrix(clusters, coarsemat)
        # find largest of diagonal value
        cmat[np.diag_indices_from(cmat)] = np.nan
        x_idxs, y_idxs = np.where(cmat == np.nanmax(cmat))

        clusters[x_idxs[0]].extend(clusters[y_idxs[0]])
        clusters = np.delete(clusters, y_idxs[0])

    return np.asarray(clusters[0], dtype=int)


@beartype
def _sort_clusters(
    clusters: Object1DArray, mat: FloatMatrix,
) -> Object1DArray:
    """Sort clusters globally by the reverse Cuthill-McKee algorithm and
    internally by the largest average values within cluster."""
    # sort the order of the cluster
    clusters_permuted = clusters[
        _sort_coarse_clustermatrix(
            _coarse_clustermatrix(clusters, mat),
        )
    ]

    # sort inside each cluster
    for cluster_idx, cluster in enumerate(clusters_permuted):
        clusters_permuted[cluster_idx] = np.array(cluster)[
            np.argsort(
                np.nanmean(mat[np.ix_(cluster, cluster)], axis=1),
            )[::-1]
        ].tolist()
    return clusters_permuted


class Clustering:
    """Class for clustering a correlation matrix.

    Parameters
    ----------
    mode : str, default='CPM'
        the mode which determines the quality function optimized by the Leiden
        algorithm.
        - 'CPM': will use the constant Potts model on the full, weighted graph
        - 'modularity': will use modularity on a knn-graph

    weighted : bool, default=True,
        If True, the underlying graph has weighted edges. Otherwise, the graph
        is constructed using the adjacency matrix.

    iterations : int, default=None,
        Number of iterations to run the Leiden algorithm. None means that the
        algorithm runs until no further improvement is achieved.

    n_neighbors: int, default=None,
        This parameter specifies if the whole matrix is used, or an knn-graph.
        The default depends on the `mode`
        - 'CPM': `None` uses full graph, and
        - 'modularity': `None` uses square root of the number of features.

    resolution_parameter : float, default=None,
        Only required for mode 'CPM'. If None, the resolution parameter will
        be set to the median value of the matrix.

    Attributes
    ----------
    clusters_ : ndarray of shape (n_clusters)
        The result of the clustering process. A list of arrays, each
        containing all indices (features) for each cluster.

    matrix_ : ndarray of shape (n_features, n_features)
        Permuted matrix according to the found clusters.

    ticks_ : ndarray of shape (n_clusters)
        Get cumulative indices where new cluster starts in `matrix_`.

    permutation_ : ndarray of shape (n_features)
        Permutation of the input features (corresponds to flattened
        `clusters_`).

    n_neighbors_ : int
        Only avaiable when using knn graph. Indicates the number of nearest
        neighbors used for constructin the knn-graph.

    resolution_param_ : float
        Only for mode 'CPM'. Indicates the resolution parameter used for the
        CPM based Leiden clustering.

    Examples
    --------
    >>> import cfs
    >>> mat = np.array([[1.0, 0.1, 0.9], [0.1, 1.0, 0.0], [0.8, 0.1, 1.0]])
    >>> clust = cfs.Clustering()
    >>> clust.fit(mat)
    >>> clust.matrix_
    array([[1. , 0.9, 0.1],
           [0.8, 1. , 0.1],
           [0.1, 0. , 1. ]])
    >>> clust.clusters_
    [array([0, 2]), array([1])]

    """

    @beartype
    def __init__(
        self,
        *,
        mode: ClusteringModeString = 'CPM',
        weighted: bool = True,
        n_neighbors: Optional[PositiveInt] = None,
        resolution_parameter: Optional[NumInRange0to1] = None,
        iterations: Optional[PositiveInt] = None,
    ) -> None:
        """Initialize Clustering class."""
        self._mode: ClusteringModeString = mode
        self._weighted: bool = weighted
        self._neighbors: Optional[PositiveInt] = n_neighbors

        self._iterations: int
        if iterations is None:
            self._iterations = -1
        else:
            self._iterations = iterations

        if mode == 'CPM':
            self._resolution_parameter: Optional[NumInRange0to1] = (
                resolution_parameter
            )
            if not weighted:
                raise NotImplementedError(
                    'mode="CPM" does not support weighted=False',
                )
        elif resolution_parameter is not None:
            raise NotImplementedError(
                'mode="modularity" does not support the usage of the '
                'resolution_parameter',
            )

    @beartype
    def fit(self, X: FloatMatrix, y: Optional[np.ndarray] = None) -> None:
        """Clusters the correlation matrix by Leiden clustering on a graph.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_features)
            Matrix containing the correlation metric which is clustered. The
            values should go from [0, 1] where 1 means completely correlated
            and 0 no correlation.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        """
        # prepare matric for graph construction
        mat: FloatMatrix
        if self._mode == 'CPM' and self._neighbors is None:
            mat = np.copy(X)
        else:
            mat = self._construct_knn_mat(X)
        # mask diagonal and zero elements
        mat[mat == 0] = np.nan
        mat[np.diag_indices_from(mat)] = np.nan

        if self._mode == 'CPM':
            if self._resolution_parameter is None:
                third_quartile = 0.75
                self._resolution_parameter = np.nanquantile(
                    mat, third_quartile,
                )

            self.resolution_param_: Optional[NumInRange0to1] = (
                self._resolution_parameter
            )

        # create graph
        mat[np.isnan(mat)] = 0
        graph: ig.Graph = ig.Graph.Weighted_Adjacency(
            list(mat.astype(np.float64)), loops=False,
        )

        clusters: Object1DArray = self._clustering_leiden(graph)
        self.clusters_: Object1DArray = _sort_clusters(clusters, X)

        self.permutation_: Index1DArray = np.hstack(self.clusters_)
        self.matrix_: Float2DArray = np.copy(X)[
            np.ix_(self.permutation_, self.permutation_)
        ]
        self.ticks_: Index1DArray = np.cumsum(
            [len(cluster) for cluster in self.clusters_],
        )

    @beartype
    def _construct_knn_mat(self, matrix: FloatMatrix) -> FloatMatrix:
        """Construct the knn matrix."""
        if self._neighbors is None:
            n_features = len(matrix)
            self._neighbors = np.floor(np.sqrt(n_features)).astype(int)
        elif self._neighbors > len(matrix):
            raise ValueError(
                'The number of nearest neighbors must be smaller than the '
                'number of features.',
            )
        self.n_neighbors_: PositiveInt = self._neighbors

        neigh = NearestNeighbors(
            n_neighbors=self._neighbors,
            metric='precomputed',
        )
        neigh.fit(1 - matrix)
        if self._weighted:
            dist_mat = neigh.kneighbors_graph(mode='distance').toarray()
            dist_mat[dist_mat == 0] = 1
            return 1 - dist_mat
        return neigh.kneighbors_graph(mode='connectivity').toarray()

    @beartype
    def _setup_leiden_kwargs(self, graph: ig.Graph) -> Dict[str, Any]:
        """Set up the parameters for the Leiden clustering."""
        kwargs_leiden = {}
        if self._iterations is None:
            kwargs_leiden['n_iterations'] = -1
        if self._mode == 'CPM':
            kwargs_leiden['partition_type'] = la.CPMVertexPartition
            kwargs_leiden[
                'resolution_parameter'
            ] = self._resolution_parameter
        else:
            kwargs_leiden[
                'partition_type'
            ] = la.ModularityVertexPartition
        if self._weighted:
            kwargs_leiden['weights'] = graph.es['weight']

        return kwargs_leiden

    @beartype
    def _clustering_leiden(self, graph: ig.Graph) -> Object1DArray:
        """Perform the Leiden clustering on the graph."""
        clusters = la.find_partition(
            graph, **self._setup_leiden_kwargs(graph),
        )
        # In case of clusters of same length, numpy casted it as a 2D array.
        # To ensure that the result is an numpy array of list, we need to
        # create an empty list, adding the values in the second step
        cluster_list = np.empty(len(clusters), dtype=object)
        cluster_list[:] = clusters  # noqa: WPS362
        return cluster_list
