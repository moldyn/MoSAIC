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
    clusters_ : list of arrays
        The result of the clustering process. A list containing of arrays,
        each containing all features for each cluster.
    """
    _kwargs_leiden = {}

    def __init__(self, *, mode='CPM', weighted=True, neighbors=None):
        """Initializes Clustering class."""
        self._mode = mode
        self._weighted = weighted
        self._neighbors = neighbors

        if self._mode == 'CPM' and not self._weighted:
            raise ValueError(
                'CPM Leiden clustering works best on a fully weighted graph.'
            )

    def fit(self, matrix, y=None):
        """Clusters the correlation matrix by Leiden clustering on a graph.

        Parameters
        ----------
        matrix : ndarray of shape (n_features, n_features)
            Matrix containing the correlation metric which is clustered.

        y : Ignored
            Not used, present for scikit API consistency by convention.
        """
        if self._mode == 'CPM':
            pass

    def _construct_knn_graph(
        self,
        matrix,
        knn=np.floor(np.sqrt(len(matrix))).astype(int),
    ):
        """Constructs the graph."""
        neigh = NearestNeighbors(
            n_neighbors=knn,
            metric='precomputed',
        )
        if not self._weighted:
            neigh.fit(1 - matrix)
        else:
            #TODO: continue

    def _construct_full_weighted_graph(self, matrix):
        """Constructs a full, weighted graph."""
        graph = ig.Graph.Weighted_Adjacency(matrix, loops=False)
        self._kwargs_leiden['weights'] = graph.es['weight']
        return graph

    def _setup_leiden_kwargs(self):
        """Sets up the parameters for the Leiden clustering"""
        if self._mode == 'CPM':
            self._kwargs_leiden['partition_type'] = la.CPMVertexPartition
            self._kwargs_leiden[
                'resolution_parameter'
            ] = self._resolution_parameter
        else:
            self._kwargs_leiden[
                'partition_type'
            ] = la.ModularityVertexPartition

    def _clustering_leiden(self, graph):
        """Performs the Leiden clustering on the graph."""
        return la.find_partition(graph, **self._kwargs_leiden)
