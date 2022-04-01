# -*- coding: utf-8 -*-
"""Class for embedding correlation matrix with UMAP.

MIT License
Copyright (c) 2021-2022, Daniel Nagel, Georg Diez
All rights reserved.

"""
__all__ = ['UMAPSimilarity']  # noqa: WPS410

import warnings

import numpy as np
import umap
from beartype import beartype
from beartype.typing import Optional

from mosaic._typing import (  # noqa:WPS436
    ArrayLikeFloat,
    Float2DArray,
    FloatMatrix,
    PositiveInt,
)


@beartype
def _calc_distance_matrix(embedding: Float2DArray) -> FloatMatrix:
    """Calculate the euclidean distance matrix from an embedding."""
    return np.sqrt(
        np.sum(
            (embedding[:, np.newaxis, :] - embedding[np.newaxis, :, :])**2,
            axis=-1,
        ),
    )


class UMAPSimilarity:  # noqa: WPS214
    r"""Class for embedding similarity matrix with UMAP.

    For more details on the parameters check the UMAP documentation.

    Parameters
    ----------
    densmap : bool, default=True
        If True the density-augmented objective of densMAP is used for
        optimization. There the local densities are encouraged to be correlated
        with those in the original space.

    n_neighbors: int, default=None
        Size of nearest neighbors used for manifold estimation in UMAP.
        If `None` uses square root of the number of features.

    n_components: int, default=2
        Dimensionality of the local embedding.

    Attributes
    ----------
    matrix_ : ndarray of shape (n_features, n_features)
        Normalized pairwise distance matrix of the UMAP embedding.

    embedding_ : ndarray of shape (n_features, n_components)
        Coordinates of features in UMAP embedding.

    n_neighbors_ : int
        Number of used neighbors.

    """

    _default_n_components: PositiveInt = 2

    @beartype
    def __init__(
        self,
        *,
        densmap: bool = True,
        n_neighbors: Optional[PositiveInt] = None,
        n_components: PositiveInt = _default_n_components,
    ):
        """Initialize UMAPSimilarity class."""
        self._densmap: bool = densmap
        self._n_neighbors: Optional[PositiveInt] = n_neighbors
        self._n_components: PositiveInt = n_components

    @beartype
    def fit(
        self,
        X: Float2DArray,
        y: Optional[ArrayLikeFloat] = None,
    ) -> None:
        """Fit similarity matrix into UMAP embedding."""
        self._reset()

        # parse data
        self._n_features: int = X.shape[0]

        # parse n_neighbors
        if self._n_neighbors is None:
            self._n_neighbors = np.ceil(np.sqrt(self._n_features)).astype(int)
        elif self._n_neighbors >= len(X):
            raise ValueError(
                'The number of nearest neighbors must be smaller than the '
                'number of features.',
            )
        self.n_neighbors_: PositiveInt = self._n_neighbors

        reducer = umap.UMAP(
            n_neighbors=self._n_neighbors,
            densmap=self._densmap,
            n_components=self._n_components,
            metric='precomputed',
        )

        # run UMAP with dissimalirty matrix
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='using precomputed metric',
            )
            embedding: np.ndarray = reducer.fit_transform(1 - X)
        matrix_: np.ndarray = _calc_distance_matrix(embedding)

        self.embedding_: np.ndarray = embedding
        self.matrix_: np.ndarray = 1 - matrix_ / np.nanmax(matrix_)

    @beartype
    def _reset(self) -> None:
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, 'matrix_'):  # noqa: WPS421
            del self.matrix_  # noqa: WPS420
