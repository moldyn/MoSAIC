# -*- coding: utf-8 -*-
"""Class for estimating correlation matrices.

MIT License
Copyright (c) 2021-2022, Daniel Nagel, Georg Diez
All rights reserved.

"""
__all__ = ['Similarity']  # noqa: WPS410

from functools import singledispatchmethod

import numpy as np
from beartype import beartype
from beartype.typing import Callable, Optional, Union
from sklearn import preprocessing
from sklearn.base import BaseEstimator

from mosaic._correlation_utils import (  # noqa: WPS436
    _correlation,
    _estimate_densities,
    _gy,
    _jsd,
    _nmi_gen,
    _nonlinear_GY_knn,
    _welford_correlation,
)
from mosaic._typing import (  # noqa: WPS436
    ArrayLikeFloat,
    Float2DArray,
    FloatMatrix,
    FloatMax2DArray,
    MetricString,
    NormString,
)


@beartype
def _standard_scaler(X: Float2DArray) -> Float2DArray:
    """Make data mean-free and std=1."""
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)


class Similarity(BaseEstimator):
    r"""Class for calculating the similarity measure.

    Parameters
    ----------
    metric : str, default='correlation'
        the correlation metric to use for the feature distance matrix.

        - `'correlation'` will use the absolute value of the Pearson
          correlation
        - `'NMI'` will use the mutual information normalized by joined entropy
        - `'GY'` uses Gel'fand and Yaglom normalization[^1]
        - `'JSD'` will use the Jensen-Shannon divergence between the joint
          probability distribution and the product of the marginal probability
          distributions to calculate their dissimilarity

        Note: `'NMI'` is supported only with low_memory=False

    low_memory : bool, default=False
        If True, the input of fit X needs to be a file name and the correlation
        is calculated on the fly. Otherwise, an array is assumed as input X.

    normalize_method : str, default='geometric'
        Only required for metric `'NMI'`. Determines the normalization factor
        for the mutual information:

        - `'joint'` is the joint entropy
        - `'max'` is the maximum of the individual entropies
        - `'arithmetic'` is the mean of the individual entropies
        - `'geometric'` is the square root of the product of the individual
          entropies
        - `'min'` is the minimum of the individual entropies

    use_knn_estimator : bool, default=False
        Can only be set for metric GY. If True, the mutual information
        is estimated reliably by a parameter free method based on entropy
        estimation from k-nearest neighbors distances[^3].
        It considerably increases the computational time and is thus
        only advisable for relatively small data-sets.

    Attributes
    ----------
    matrix_ : ndarray of shape (n_features, n_features)
        The correlation-measure-based pairwise distance matrix of the data. It
        scales from [0, 1].

    Examples
    --------
    >>> import mosaic
    >>> x = np.linspace(0, np.pi, 1000)
    >>> data = np.array([np.cos(x), np.cos(x + np.pi / 6)]).T
    >>> sim = mosaic.Similarity()
    >>> sim.fit(data)
    Similarity()
    >>> sim.matrix_
    array([[1.       , 0.9697832],
           [0.9697832, 1.       ]])


    Notes
    -----
    The correlation is defined as
    $$\rho_{X,Y} =
    \frac{\langle(X -\mu_X)(Y -\mu_Y)\rangle}{\sigma_X\sigma_Y}$$
    where for the online (low memory) algorithm the Welford algorithm taken
    from Donald E. Knuth were used [^2].

    [^1]: Gel'fand, I.M. and Yaglom, A.M. (1957). "Calculation of amount of
        information about a random function contained in another such
        function".
        American Mathematical Society Translations, series 2, 12, pp. 199–246.

    [^2]: Welford algorithm, generalized to correlation. Taken from:
        Donald E. Knuth (1998). "The Art of Computer Programming", volume 2:
        Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

    [^3]: B.C. Ross, PLoS ONE 9(2) (2014), "Mutual Information between Discrete
        and Continuous Data Sets"

    The Jensen-Shannon divergence is defined as
    $$D_{\text{JS}} = \frac{1}{2} D_{\text{KL}}(p(x,y)||M)
    + \frac{1}{2} D_{\text{KL}}(p(x)p(y)||M)\;,$$
    where \(M = \frac{1}{2} [p(x,y) + p(x)p(y)]\) is an averaged probability
    distribution and \(D_{\text{KL}}\) denotes the Kullback-Leibler divergence.

    """

    _dtype: np.dtype = np.float64
    _default_normalize_method: str = 'geometric'

    @beartype
    def __init__(
        self,
        *,
        metric: MetricString = 'correlation',
        low_memory: bool = False,
        normalize_method: Optional[NormString] = None,
        use_knn_estimator: bool = False,
    ):
        """Initialize Similarity class."""
        self.metric: MetricString = metric
        self.low_memory: bool = low_memory
        self.use_knn_estimator: bool = use_knn_estimator
        if self.metric == 'NMI':
            if normalize_method is None:
                normalize_method = self._default_normalize_method
        elif normalize_method is not None:
            raise NotImplementedError(
                'Normalize methods are only supported with metric="NMI"',
            )
        self.normalize_method: NormString = normalize_method
        if self.metric != 'GY' and self.use_knn_estimator:
            raise NotImplementedError(
                (
                    'The mutual information estimate based on k-nearest'
                    'neighbors distances is only supported with metric="GY"'
                ),
            )

    @singledispatchmethod
    @beartype
    def fit(
        self,
        X: Union[FloatMax2DArray, str],
        y: Optional[ArrayLikeFloat] = None,
    ):
        """Compute the correlation/nmi distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if low_memory=True
            Training data.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        raise NotImplementedError('Fatal error, this should never be reached.')

    @fit.register
    def _(self, X: np.ndarray, y=None):
        """Dispatched for low_memory=False with matrix input."""
        self._reset()

        corr: np.ndarray
        matrix_: np.ndarray
        # parse data
        if self.low_memory:
            raise TypeError('Using low_memory=True requires X:str')
        if X.ndim == 1:
            raise ValueError(
                'Reshape your data either using array.reshape(-1, 1) if your '
                'data has a single feature or array.reshape(1, -1) if it '
                'contains a single sample.',
            )

        n_samples, n_features = X.shape
        self._n_samples: int = n_samples
        self._n_features: int = n_features

        X: np.ndarray = _standard_scaler(X)
        if self.metric == 'correlation':
            corr = _correlation(X)
            matrix_ = np.abs(corr)
        elif self.metric == 'GY' and self.use_knn_estimator:
            matrix_ = _nonlinear_GY_knn(X)
        else:  # 'NMI', 'JSD', 'GY
            matrix_ = self._nonlinear_correlation(X)
        self.matrix_: np.ndarray = np.clip(matrix_, a_min=0, a_max=1)

        return self

    @fit.register
    def _(self, X: str, y=None):
        """Dispatched for low_memory=True with string."""
        self._reset()

        corr: np.ndarray
        matrix_: np.ndarray
        # parse data
        if not self.low_memory:
            raise TypeError('Mode low_memory=False reuqires X:np.ndarray.')

        if self.metric == 'correlation':
            corr = self._online_correlation(X)
            matrix_ = np.abs(corr)
        else:
            raise NotImplementedError(
                'Mode low_memory=True is only implemented for correlation.',
            )

        self.matrix_: np.ndarray = np.clip(matrix_, a_min=0, a_max=1)

        return self

    @beartype
    def fit_transform(
        self,
        X: Union[FloatMax2DArray, str],
        y: Optional[ArrayLikeFloat] = None,
    ) -> FloatMatrix:
        """Compute the correlation/nmi distance matrix and returns it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if low_memory=True
            Training data.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        Similarity : ndarray of shape (n_features, n_features)
            Similarity matrix.

        """
        self.fit(X)
        return self.matrix_

    @beartype
    def transform(
        self,
        X: Union[FloatMax2DArray, str],
    ) -> FloatMatrix:
        """Compute the correlation/nmi distance matrix and returns it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if low_memory=True
            Training data.

        Returns
        -------
        Similarity : ndarray of shape (n_features, n_features)
            Similarity matrix.

        """
        return self.fit_transform(X)

    @beartype
    def _reset(self) -> None:
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, 'matrix_'):  # noqa: WPS421
            del self.matrix_  # noqa: WPS420

    @beartype
    def _nonlinear_correlation(self, X: Float2DArray) -> FloatMatrix:
        """Return the nonlinear correlation."""
        calc_nl_corr: Callable
        if self.metric == 'NMI':
            calc_nl_corr = _nmi_gen(self.normalize_method)
        elif self.metric == 'GY':
            calc_nl_corr = _gy
        else:
            calc_nl_corr = _jsd

        nl_corr: FloatMatrix = np.empty(  # noqa: WPS317
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        for idx_i, xi in enumerate(X.T):
            nl_corr[idx_i, idx_i] = 1
            for idx_j, xj in enumerate(X.T[idx_i + 1:], idx_i + 1):
                nl_corr_ij = calc_nl_corr(*_estimate_densities(xi, xj))
                nl_corr[idx_i, idx_j] = nl_corr_ij
                nl_corr[idx_j, idx_i] = nl_corr_ij

        return nl_corr

    @beartype
    def _online_correlation(self, X: str) -> FloatMatrix:
        """Calculate correlation on the fly."""
        self._filename: str = X

        corr, n_samples, n_features = _welford_correlation(
            filename=self._filename,
            dtype=self._dtype,
        )
        self._n_samples: int = n_samples
        self._n_features: int = n_features
        return corr
