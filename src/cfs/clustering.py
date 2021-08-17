# -*- coding: utf-8 -*-
"""Class for clustering the correlation matrices.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
from typing import Any, Dict, Optional

import igraph as ig
import leidenalg as la
import numpy as np
from beartype import beartype
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors


@beartype
def _coarse_clustermatrix(
    clusters: np.ndarray, mat: np.ndarray,
) -> np.ndarray:
    """Construct a coarse cluster matrix by averaging over all clusters."""
    if len(clusters) == len(mat):
        return np.copy(mat)

    nclusters = len(clusters)
    return np.array([
        mat[np.ix_(clusters[idx_i], clusters[idx_j])].mean()
        for idx_i, idx_j in np.ndindex(nclusters, nclusters)
    ]).reshape(nclusters, nclusters)


def _cuthill_mckee_sorting(coarsemat):
    """Resort clusters to minimize off-diagonal distances."""
    nclusters = len(coarsemat)
    if nclusters > 1:
        scale_factor = int(np.ceil(np.sqrt(nclusters)))
        cutoff = nclusters**2 - scale_factor * nclusters
    else:
        raise ValueError(
            'Only one cluster was found. Try different parameters',
        )
    coarsemat[
        coarsemat < np.sort(coarsemat, axis=None)[cutoff]
    ] = np.nan
    rCMcKee = csgraph.csgraph_from_dense(coarsemat)
    return csgraph.reverse_cuthill_mckee(
        rCMcKee,
        symmetric_mode=True,
    )


@beartype
def _sort_clusters(
    clusters: np.ndarray, mat: np.ndarray,
) -> np.ndarray:
    """Sort clusters globally by the reverse Cuthill-McKee algorithm and
    internally by the largest average values within cluster."""
    clusters_permuted = clusters[
        _cuthill_mckee_sorting(
            _coarse_clustermatrix(clusters, mat),
        )
    ]

    return np.array(
        [
            np.array(cluster)[
                np.argsort(
                    np.nanmean(mat[np.ix_(cluster, cluster)], axis=1),
                )[::-1]
            ]
            for cluster in clusters_permuted
        ],
        dtype=object,
    )


class Clustering:
    """Class for clustering a correlation matrix.

    Parameters
    ----------
    mode : str, default='CPM'
        the mode which determines the quality function optimized by the Leiden
        algorithm.

        - 'CPM' will use the constant Potts model on the full, weighted graph
        - 'modularity' will use modularity on a knn-graph

    weighted : bool, default=True,
        If True, the underlying graph has weighted edges. Otherwise, the graph
        is constructed using the adjacency matrix.

    iterations : int, default=-1,
        Number of iterations to run the Leiden algorithm. -1 means that the
        algorithm runs until no further improvement is achieved.

    neighbors: int, default=None,
        Only required for mode 'modularity'. If None, the number of neighbors
        is chosen as the square root of the number of features.

    resolution_parameter : float, default=None,
        Only required for mode 'CPM'. If None, the resolution parameter will
        be set to the median value of the matrix.

    Attributes
    ----------
    clusters_ : list of ndarrays
        The result of the clustering process. A list of arrays, each
        containing all indices (features) for each cluster.

    matrix_ : ndarray of shape (n_features, n_features)
        Permuted matrix according to the found clusters.

    ticks_ : ndarray of shape (n_clusters)
        Get cumulative indices where new cluster starts in `matrix_`.

    permutation_ : ndarray of shape (n_features)
        Permutation of the input features (corresponds to flattened
        `clusters_`).

    nneighbors_ : int
        Only for mode 'modularity'. Indicates the number of nearest neighbors
        used for constructin the knn-graph.

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

    _available_modes = ('CPM', 'modularity')

    @beartype
    def __init__(
        self,
        *,
        mode: str = 'CPM',
        weighted: bool = True,
        neighbors: int = None,
        resolution_parameter: float = None,
        iterations: int = None,
    ) -> None:
        """Initialize Clustering class."""
        if mode not in self._available_modes:
            modes = ', '.join([f'"{m}"' for m in self._available_modes])
            raise NotImplementedError(
                f'Mode {mode} is not implemented, use one of [{modes}].',
            )

        self._mode = mode
        self._weighted = weighted
        self._neighbors = neighbors
        self._resolution_parameter = resolution_parameter
        self._iterations = iterations

        if self._mode == 'CPM' and not self._weighted:
            raise NotImplementedError(
                'mode="CPM" does not support an unweighted=True.',
            )

    @beartype
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
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
        if self._mode == 'CPM':
            mat = np.copy(X)
            if self._resolution_parameter is None:
                self._resolution_parameter = np.median(mat)
        elif self._mode == 'modularity':
            mat = self._construct_knn_mat(X)
        else:
            raise NotImplementedError(
                f'Mode {self._mode} is not implemented',
            )
        mat[np.isnan(mat)] = 0
        graph = ig.Graph.Weighted_Adjacency(
            list(mat.astype(np.float64)), loops=False,
        )
        clusters = self._clustering_leiden(graph)
        self.clusters_ = _sort_clusters(clusters, X)

        self.permutation_ = np.hstack(self.clusters_)
        self.matrix_ = X[np.ix_(self.permutation_, self.permutation_)]
        self.ticks_ = np.cumsum([len(cluster) for cluster in self.clusters_])
        self.resolution_param_ = self._resolution_parameter

    @beartype
    def _construct_knn_mat(self, matrix: np.ndarray) -> np.ndarray:
        """Construct the knn matrix."""
        if self._neighbors is None:
            n_features = len(matrix)
            self._neighbors = np.floor(np.sqrt(n_features)).astype(int)
            self.nneighbors_ = self._neighbors
        elif self._neighbors > len(matrix):
            raise ValueError(
                'The number of nearest neighbors must be smaller than the '
                'number of features.',
            )
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
    def _setup_leiden_kwargs(self, graph) -> Dict[str, Any]:
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
    def _clustering_leiden(self, graph: ig.Graph) -> np.ndarray:
        """Perform the Leiden clustering on the graph."""
        clusters = la.find_partition(
            graph, **self._setup_leiden_kwargs(graph),
        )
        return np.array(list(clusters), dtype=object)
