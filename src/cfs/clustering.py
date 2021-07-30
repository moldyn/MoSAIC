# -*- coding: utf-8 -*-
"""Class for clustering the correlation matrices.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.neighbors import NearestNeighbors


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

    neighbors: int, default=None,
        Only required for mode 'modularity'. If NaN, the number of neighbors
        is chosen as the square root of the number of features.

    resolution_parameter : float, default= 0.7
        Only required for mode 'CPM'

    Attributes
    ----------
    clusters_ : list of ndarrays
        The result of the clustering process. A list of arrays, each containing
        all indices (features) for each cluster.

    """
    _available_modes = ('CPM', 'modularity')

    def __init__(
        self,
        *,
        mode='CPM',
        weighted=True,
        neighbors=None,
        resolution_parameter=0.7,
    ):
        """Initializes Clustering class."""
        if mode not in self._available_modes:
            raise NotImplementedError(
                f'Mode {mode} is not implemented, use one of ['
                f'{" ".join(self._available_modes)}].',
            )

        self._mode = mode
        self._weighted = weighted
        self._neighbors = neighbors
        self._resolution_parameter = resolution_parameter

        if self._mode == 'CPM' and not self._weighted:
            raise NotImplementedError(
                'mode="CPM" does not support an unweighted=True.'
            )

    def fit(self, matrix, y=None):
        """Clusters the correlation matrix by Leiden clustering on a graph.

        Parameters
        ----------
        matrix : ndarray of shape (n_features, n_features)
            Matrix containing the correlation metric which is clustered. The
            values should go from [0, 1] where 1 means completely correlated
            and 0 no correlation.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        """
        if self._mode == 'CPM':
            mat = np.copy(matrix)
        elif self._mode == 'modularity':
            mat = self._construct_knn_mat(matrix)
        else:
            raise NotImplementedError(
                f'Mode {self._mode} is not implemented',
            )
        mat[np.isnan(mat)] = 0
        graph = ig.Graph.Weighted_Adjacency(
            mat.astype(np.float64), loops=False,
        )
        clusters = self._clustering_leiden(graph)
        self.clusters_ = self._sort_clusters(clusters, matrix)

    def _construct_knn_mat(self, matrix):
        """Constructs the knn matrix."""
        if self._neighbors is None:
            self._neighbors = np.floor(np.sqrt(len(matrix))).astype(int)
        neigh = NearestNeighbors(
            n_neighbors=self._neighbors,
            metric='precomputed',
        )
        neigh.fit(1 - matrix)
        if self._weighted:
            dist_mat = neigh.kneighbors_graph(mode='distance').toarray()
            dist_mat[dist_mat == 0] = 1
            mat = 1 - dist_mat
        else:
            mat = neigh.kneighbors_graph(mode='connectivity').toarray()

        return mat

    def _setup_leiden_kwargs(self, graph):
        """Sets up the parameters for the Leiden clustering"""
        kwargs_leiden = {}
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

    def _clustering_leiden(self, graph):
        """Perform the Leiden clustering on the graph."""
        return la.find_partition(
            graph, **self._setup_leiden_kwargs(graph),
        )

    def _sort_clusters(self, clusters, mat):
        """Sort clusters by largest average values within cluster."""
        sorted_clusters = []
        for cluster in clusters:
            perm = np.argsort(
                np.nanmean(mat[np.ix_(cluster, cluster)], axis=1)
            )[::-1]
            sorted_clusters.append(
                np.array(cluster)[perm],
            )
        return sorted_clusters
