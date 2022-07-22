# -*- coding: utf-8 -*-
"""Class for clustering the correlation matrices.

MIT License
Copyright (c) 2021-2022, Daniel Nagel, Georg Diez
All rights reserved.

"""
__all__ = ['Clustering']  # noqa: WPS410

import igraph as ig
import leidenalg as la
import numpy as np
from beartype import beartype
from beartype.typing import Any, Dict, Optional
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted
from sklearn_extra.cluster import KMedoids

from mosaic._typing import (  # noqa: WPS436
    ClusteringModeString,
    Float,
    Float2DArray,
    FloatMatrix,
    Index1DArray,
    Int,
    NumInRange0to1,
    Object1DArray,
    PositiveInt,
    SimilarityMatrix,
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
def _sort_coarse_clustermatrix(
    clusterssize: Object1DArray, coarsemat: FloatMatrix,
) -> Index1DArray:
    """Return indices which sort clusters to minimize off-diagonal values."""
    nclusters = len(coarsemat)
    clusters = np.empty(nclusters, dtype=object)
    clusters[:] = [[i] for i in range(nclusters)]  # noqa: WPS362
    # make deep copy of clusterssize
    coarseclusters = _copy_clusters(clusterssize)

    for _ in range(nclusters - 1):
        cmat = _coarse_clustermatrix(clusters, coarsemat)
        # find largest of diagonal value
        cmat[np.diag_indices_from(cmat)] = np.nan
        (xi, *_), (yi, *_) = np.where(cmat == np.nanmax(cmat))

        # add smaller to larger cluster
        x_large, x_small = (xi, yi) if (
            len(coarseclusters[xi]) >= len(coarseclusters[yi])
        ) else (yi, xi)

        # merge clusters
        clusters = _merge_clusters(clusters, x_large, x_small)
        # keep track of clusrer size
        coarseclusters = _merge_clusters(coarseclusters, x_large, x_small)

    return np.asarray(clusters[0], dtype=int)


@beartype
def _merge_clusters(
    clusters: Object1DArray, idx_i: Int, idx_j: Int,
) -> Object1DArray:
    clusters[idx_i].extend(clusters[idx_j])
    return np.delete(clusters, idx_j)


@beartype
def _copy_clusters(clusters: Object1DArray) -> Object1DArray:
    new_clusters = np.empty(len(clusters), dtype=object)
    new_clusters[:] = [  # noqa: WPS362
        list(cl) for cl in clusters
    ]
    return new_clusters


@beartype
def _sort_clusters(
    clusters: Object1DArray, mat: FloatMatrix,
) -> Object1DArray:
    """Sort clusters globally and internally.

    Both are sorted by the largest average values within/between cluster,
    respectively.

    """
    # sort the order of the cluster
    clusters_permuted = clusters[
        _sort_coarse_clustermatrix(
            clusters, _coarse_clustermatrix(clusters, mat),
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


class Clustering(ClusterMixin, BaseEstimator):
    r"""Class for clustering a correlation matrix.

    Parameters
    ----------
    mode : str, default='CPM'
        the mode which determines the quality function optimized by the Leiden
        algorithm ('CPM', or 'modularity') or linkage clustering.
        - 'CPM': will use the constant Potts model on the full, weighted graph
        - 'modularity': will use modularity on a knn-graph
        - 'linkage': will use complete-linkage clustering
        - 'kmedoids': will use k-medoids clustering

    weighted : bool, default=True
        If True, the underlying graph has weighted edges. Otherwise, the graph
        is constructed using the adjacency matrix.

    n_neighbors : int, default=None
        This parameter specifies whether the whole matrix should be used, or
        a knn-graph, which reduces the required memory.
        The default depends on the `mode`
        - 'CPM': `None` uses the full graph, and
        - 'modularity': `None` uses square root of the number of features.

    resolution_parameter : float, default=None
        Required for mode 'CPM' and 'linkage'. If None, the resolution
        parameter will be set to the third quartile of `X` for
        `n_neighbors=None` and else to the mean value of the knn graph.

    n_clusters : int, default=None
        Required for 'kmedoids'. The number of medoids which will constitute
        the later clusters.

    seed : int, default=None
        Use an integer to make the randomness of Leidenalg deterministic. By
        default uses a random seed if nothing is specified.

    Attributes
    ----------
    clusters_ : ndarray of shape (n_clusters, )
        The result of the clustering process. A list of arrays, each
        containing all indices (features) corresponging to each cluster.

    labels_ : ndarray of shape (n_features, )
        Labels of each feature.

    matrix_ : ndarray of shape (n_features, n_features)
        Permuted matrix according to the determined clusters.

    ticks_ : ndarray of shape (n_clusters, )
        The cumulative number of features containing to the clusters.
        May be used as ticks for plotting `matrix_`.

    permutation_ : ndarray of shape (n_features, )
        Permutation of the input features (corresponds to flattened
        `clusters_`).

    n_neighbors_ : int
        Only avaiable when using knn graph. Indicates the number of nearest
        neighbors used for constructin the knn-graph.

    resolution_param_ : float
        Only for mode 'CPM' and 'linkage'. Indicates the resolution parameter
        used for the CPM based Leiden clustering.

    linkage_matrix_ : ndarray of shape (n_clusters - 1, 4)
        Only for mode 'linkage'. Contains the hierarchical clustering encoded
        as a linkage matrix, see
        [scipy:spatial.distance.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

    Examples
    --------
    >>> import mosaic
    >>> mat = np.array([[1.0, 0.1, 0.9], [0.1, 1.0, 0.1], [0.9, 0.1, 1.0]])
    >>> clust = mosaic.Clustering()
    >>> clust.fit(mat)
    Clustering(resolution_parameter=0.7)
    >>> clust.matrix_
    array([[1. , 0.9, 0.1],
           [0.9, 1. , 0.1],
           [0.1, 0.1, 1. ]])
    >>> clust.clusters_
    array([list([2, 0]), list([1])], dtype=object)

    """

    @beartype
    def __init__(
        self,
        *,
        mode: ClusteringModeString = 'CPM',
        weighted: bool = True,
        n_neighbors: Optional[PositiveInt] = None,
        resolution_parameter: Optional[NumInRange0to1] = None,
        n_clusters: Optional[PositiveInt] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Clustering class."""
        self.mode: ClusteringModeString = mode
        self.n_clusters: Optional[PositiveInt] = n_clusters
        self.n_neighbors: Optional[PositiveInt] = n_neighbors
        self.resolution_parameter: Optional[NumInRange0to1] = (
            resolution_parameter
        )
        self.seed: Optional[int] = seed
        self.weighted: bool = weighted

        if mode in {'linkage', 'kmedoids'} and self.n_neighbors is not None:
            raise NotImplementedError(
                f"mode='{mode}' does not support knn-graphs.",
            )

        if mode == 'kmedoids' and self.n_clusters is None:
            raise TypeError(
                f"mode='{mode}' needs parameter 'n_clusters'",
            )
        elif mode != 'kmedoids' and self.n_clusters is not None:
            raise NotImplementedError(
                f"mode='{mode}' does not support the usage of 'n_clusters'",
            )

        if mode in {'CPM', 'linkage'}:
            if not weighted:
                raise NotImplementedError(
                    f"mode='{mode}' does not support weighted=False",
                )
        elif resolution_parameter is not None:
            raise NotImplementedError(
                f"mode='{mode}' does not support the usage of the "
                'resolution_parameter',
            )

    @beartype
    def fit(self, X: SimilarityMatrix, y: Optional[np.ndarray] = None):
        """Clusters the correlation matrix by Leiden clustering on a graph.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_features)
            Matrix containing the correlation metric which is clustered. The
            values should go from [0, 1] where 1 means completely correlated
            and 0 no correlation.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        self._reset()

        # prepare matric for graph construction
        mat: FloatMatrix
        if self.mode in {'linkage', 'kmedoids'}:
            mat = np.copy(X)
        elif self.mode == 'CPM' and self.n_neighbors is None:
            mat = np.copy(X)
        else:
            mat = self._construct_knn_mat(X)

        if self.mode in {'CPM', 'linkage'}:
            # mask diagonal and zero elements
            mat[mat == 0] = np.nan
            mat[np.diag_indices_from(mat)] = np.nan

            if self.resolution_parameter is None:
                if self.n_neighbors is None:
                    third_quartile = 0.75
                    self.resolution_parameter = np.nanquantile(
                        mat, third_quartile,
                    )
                else:
                    self.resolution_parameter = np.nanmean(mat)

            self.resolution_param_: NumInRange0to1 = (
                self.resolution_parameter
            )

        # create graph
        mat[np.isnan(mat)] = 0

        clusters: Object1DArray
        if self.mode == 'linkage':
            clusters = self._clustering_linkage(mat)
        elif self.mode == 'kmedoids':
            clusters = self._clustering_kmedoids(mat)
        else:  # _mode in {'CPM', 'modularity'}
            graph: ig.Graph = ig.Graph.Weighted_Adjacency(
                list(mat.astype(np.float64)), loops=False,
            )
            clusters = self._clustering_leiden(graph)

        self.clusters_: Object1DArray = _sort_clusters(clusters, X)
        self.permutation_: Index1DArray = np.hstack(self.clusters_)
        self.matrix_: Float2DArray = np.copy(X)[
            np.ix_(self.permutation_, self.permutation_)
        ]
        self.ticks_: Index1DArray = np.cumsum(
            [len(cluster) for cluster in self.clusters_],
        )
        labels: Index1DArray = np.empty_like(self.permutation_)
        for idx, cluster in enumerate(self.clusters_):
            labels[cluster] = idx
        self.labels_: Index1DArray = labels

        return self

    @beartype
    def fit_predict(
        self, X: SimilarityMatrix, y: Optional[np.ndarray] = None,
    ) -> Index1DArray:
        """Clusters the correlation matrix by Leiden clustering on a graph.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_features)
            Matrix containing the correlation metric which is clustered. The
            values should go from [0, 1] where 1 means completely correlated
            and 0 no correlation.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.

        """
        return super().fit_predict(X, y)

    @beartype
    def score(
        self,
        X: SimilarityMatrix,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Float:
        """Estimate silhouette_score of new correlation matrix.

        Parameters
        ----------
        X : ndarray of shape (n_features, n_features)
            New matrix containing the correlation metric to score. The
            values should go from [0, 1] where 1 means completely correlated
            and 0 no correlation.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        sample_weight: Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        score : float
            Silhouette score of new correlation matrix based on fitted labels.

        """
        check_is_fitted(self, attributes=['labels_', 'matrix_'])

        n_labels = len(self.labels_)
        n_unique_labels = len(np.unique(self.labels_))

        if n_labels != len(X):
            raise ValueError(
                f'Dimension of X d={len(X):.0f} needs to agree with the '
                f'dimension of the fitted data d={n_labels:.0f}.',
            )

        if n_unique_labels in {1, n_labels}:
            return -1.0
        return silhouette_score(X, labels=self.labels_)

    @beartype
    def _reset(self) -> None:
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, 'clusters_'):  # noqa: WPS421
            del self.clusters_  # noqa: WPS420
            del self.labels_  # noqa: WPS420
            del self.ticks_  # noqa: WPS420
            del self.permutation_  # noqa: WPS420
            del self.matrix_  # noqa: WPS420

        if hasattr(self, 'linkage_matrix_'):  # noqa: WPS421
            del self.linkage_matrix_  # noqa: WPS420
        if hasattr(self, 'n_neighbors_'):  # noqa: WPS421
            del self.n_neighbors_  # noqa: WPS420
        if hasattr(self, 'resolution_param_'):  # noqa: WPS421
            del self.resolution_param_  # noqa: WPS420
        if hasattr(self, 'n_clusters_'):  # noqa: WPS421
            del self.n_clusters_  # noqa: WPS420

    @beartype
    def _construct_knn_mat(self, matrix: FloatMatrix) -> FloatMatrix:
        """Construct the knn matrix."""
        if self.n_neighbors is None:
            n_features = len(matrix)
            self.n_neighbors = np.ceil(np.sqrt(n_features)).astype(int)
        elif self.n_neighbors >= len(matrix):
            raise ValueError(
                'The number of nearest neighbors must be smaller than the '
                'number of features.',
            )
        self.n_neighbors_: PositiveInt = self.n_neighbors

        neigh = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric='precomputed',
        )
        neigh.fit(1 - matrix)
        if self.weighted:
            dist_mat = neigh.kneighbors_graph(mode='distance').toarray()
            dist_mat[dist_mat == 0] = 1
            return 1 - dist_mat
        return neigh.kneighbors_graph(mode='connectivity').toarray()

    @beartype
    def _setup_leiden_kwargs(self, graph: ig.Graph) -> Dict[str, Any]:
        """Set up the parameters for the Leiden clustering."""
        kwargs_leiden = {'n_iterations': -1}
        if self.mode == 'CPM':
            kwargs_leiden['partition_type'] = la.CPMVertexPartition
            kwargs_leiden[
                'resolution_parameter'
            ] = self.resolution_param_
        else:
            kwargs_leiden[
                'partition_type'
            ] = la.ModularityVertexPartition
        if self.weighted:
            kwargs_leiden['weights'] = graph.es['weight']

        kwargs_leiden['seed'] = self.seed

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
        cluster_list: Object1DArray = np.empty(len(clusters), dtype=object)
        cluster_list[:] = clusters  # noqa: WPS362
        return cluster_list

    @beartype
    def _clustering_linkage(self, matrix: FloatMatrix) -> Object1DArray:
        """Perform the linkage clustering."""
        matrix[np.diag_indices_from(matrix)] = 1
        linkage_matrix: Float2DArray = linkage(
            squareform(1 - matrix),
            method='complete',
            optimal_ordering=True,
        )
        # store linkage tree
        self.linkage_matrix_: Float2DArray = linkage_matrix

        cuttree: Index1DArray = cut_tree(
            linkage_matrix, height=1 - self.resolution_param_,
        ).flatten()

        # In case of clusters of same length, numpy casted it as a 2D array.
        # To ensure that the result is an numpy array of list, we need to
        # create an empty list, adding the values in the second step
        nclusters: int = len(np.unique(cuttree))
        cluster_list: Object1DArray = np.empty(nclusters, dtype=object)
        cluster_list[:] = [  # noqa: WPS362
            np.where(cuttree == cluster)[0].tolist()
            for cluster in np.unique(cuttree)
        ]
        return cluster_list

    @beartype
    def _clustering_kmedoids(self, matrix: FloatMatrix) -> Object1DArray:
        """Perform k-medoids clustering."""
        kmedoids_kwargs = {
            'metric': 'precomputed',
            'max_iter': 100000,
            'method': 'pam',
        }

        kmedoids = KMedoids(n_clusters=self.n_clusters, **kmedoids_kwargs)
        kmedoids.fit(1 - matrix)
        labels = kmedoids.labels_

        # store number of clusters
        nclusters: int = len(np.unique(labels))
        if nclusters != self.n_clusters:
            raise ValueError(
                f'k-medoids tried to find {self.n_clusters} clusters'
                f'but only {nclusters} found. Please try a different value.',
            )
        self.n_clusters_: Float2DArray = nclusters

        # In case of clusters of same length, numpy casted it as a 2D array.
        # To ensure that the result is an numpy array of list, we need to
        # create an empty list, adding the values in the second step
        cluster_list: Object1DArray = np.empty(nclusters, dtype=object)
        cluster_list[:] = [  # noqa: WPS362
            np.where(labels == label)[0].tolist()
            for label in np.unique(labels)
        ]
        return cluster_list
