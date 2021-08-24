# -*- coding: utf-8 -*-
"""Class for embedding correlation matrix with UMAP.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
__all__ = ['UMAPSimilarity']  # noqa: WPS410

from typing import Optional

import numpy as np
import umap
from beartype import beartype

from cfs.typing import (
    ArrayLikeFloat,
    Float2DArray,
    FloatMatrix,
    PositiveInt,
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

    n_neighbors: int, default=10
        Size of nearest neighbors used for manifold estimation in UMAP.

    n_components: int, default=2
        Dimensionality of the local embedding.

    Attributes
    ----------
    matrix_ : ndarray of shape (n_features, n_features)
        Normalized pairwise distance matrix of the UMAP embedding.

    embedding_ : ndarray of shape (n_features, n_components)
        Coordinates of features in UMAP embedding.

    """

    default_n_neighbors: PositiveInt = 10
    default_n_components: PositiveInt = 2

    @beartype
    def __init__(
        self,
        *,
        densmap: bool = True,
        n_neighbors: PositiveInt = default_n_neighbors,
        n_components: PositiveInt = default_n_components,
    ):
        """Initialize UMAPSimilarity class."""
        self._densmap: bool = densmap
        self._n_neighbors: PositiveInt = n_neighbors
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

        reducer = umap.UMAP(
            n_neighbors=self._n_neighbors,
            densmap=self._densmap,
            n_components=self._n_components,
        )

        # run UMAP with dissimalirty matrix
        embedding: np.ndarray = reducer.fit_transform(1 - X)
        matrix_: np.ndarray = self._calc_distance_matrix(embedding)

        self.embedding_: np.ndarray = embedding
        self.matrix_: np.ndarray = 1 - matrix_ / np.nanmax(matrix_)

    @beartype
    def _reset(self) -> None:
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, 'matrix_'):  # noqa: WPS421
            del self.matrix_  # noqa: WPS420

    @beartype
    def _calc_distance_matrix(self, embedding: Float2DArray) -> FloatMatrix:
        """Calculate the euclidean distance matrix."""
        return np.sqrt(
            np.sum(
                (embedding[:, np.newaxis, :] - embedding[np.newaxis, :, :])**2,
                axis=-1,
            ),
        )
