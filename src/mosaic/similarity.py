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
from beartype.typing import Callable, Generator, Optional, Tuple, Union
from scipy.spatial.distance import jensenshannon
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_regression

from mosaic._typing import (  # noqa: WPS436
    ArrayLikeFloat,
    Float1DArray,
    Float2DArray,
    FloatMatrix,
    FloatMax2DArray,
    MetricString,
    NormString,
    PositiveInt,
)


@beartype
def _freedman_diaconis_rule(x: Float1DArray) -> int:
    """Freedman Diaconis rule to estimate number of bins.

    We replaced the prefactor 2 by 2.59 to asymptotically match Scott's normal
    reference rule for a normal distribution. See
    https://www.wikiwand.com/en/Freedman–Diaconis_rule

    Freedman, David and Diaconis, Persi (1981):
    "On the histogram as a density estimator: L2 theory"
    Probability Theory and Related Fields. 57 (4), 453–476

    """
    scott_factor = 2.59
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    binwidth = scott_factor * iqr / np.cbrt(len(x))
    ptp = np.min([
        np.subtract(*np.percentile(x, [85, 0])),
        np.subtract(*np.percentile(x, [100, 15])),
    ])
    return int(
        np.ceil(ptp / binwidth),
    )


@beartype
def _entropy(p: ArrayLikeFloat) -> float:
    """Calculate entropy of density p."""
    return -1 * np.sum(p * np.ma.log(p))


@beartype
def _kullback(p: ArrayLikeFloat, q: ArrayLikeFloat) -> float:
    if len(p) != len(q):
        raise ValueError(
            f'Arrays p, q need to be of same length, but {len(p):.0f} vs '
            f'{len(q):.0f}.',
        )
    """Calculate Kullback-Leibler divergence of density p, q."""
    return np.sum(
        p * np.ma.log(np.ma.divide(p, q)),
    )


@beartype
def _standard_scaler(X: Float2DArray) -> Float2DArray:
    """Make data mean-free and std=1."""
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)


@beartype
def _estimate_densities(
    x: Float1DArray, y: Float1DArray, bins: Optional[PositiveInt] = None,
) -> Tuple[Float2DArray, Float2DArray, Float1DArray, Float1DArray]:
    """Calculate two dimensional probability densities."""
    if bins is None:
        bins = [
            _freedman_diaconis_rule(x),
            _freedman_diaconis_rule(y),
        ]
    hist, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    # transpose since numpy considers axis 0 as y and axis 1 as x
    pxy = hist.T / np.sum(hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    pxpy = px[:, np.newaxis] * py[np.newaxis, :]

    return pxy, pxpy, px, py


@beartype
def _correlation(X: Float2DArray) -> FloatMatrix:
    """Return the correlation of input.

    Each feature (column) of X need to be mean-free with standard deviation 1.

    """
    return X.T / len(X) @ X


@beartype
def _knn_mutual_information(
    X: Float2DArray,
    nfeatures: PositiveInt,
) -> FloatMatrix:
    """Return the knn-estimated mutual information of input."""
    mi_knn = np.empty((nfeatures, nfeatures))
    for numf, feature in enumerate(X.T):
        mi_knn[numf] = mutual_info_regression(
            X, feature,
        )
    return mi_knn


class Similarity:  # noqa: WPS214
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

        Note: `'NMI'` is supported only with online=False

    online : bool, default=False
        If True, the input of fit X needs to be a file name and the correlation
        is calculated on the fly. Otherwise, an array is assumed as input X.

    normalize_method : str, default='geometric'
        Only required for metric `'NMI'`. Determines the normalization factor
        for the mutual information:

        - `'joint'` is the joint entropy
        - `'max'`is the maximum of the individual entropies
        - `'arithmetic'` is the mean of the individual entropies
        - `'geometric'` is the square root of the product of the individual
          entropies
        - `'min'` is the minimum of the individual entropies

    knn_estimator : bool, default=False
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
    >>> sim.matrix_
    array([[1.       , 0.9697832],
           [0.9697832, 1.       ]])


    Notes
    -----
    The correlation is defined as
    $$\rho_{X,Y} =
    \frac{\langle(X -\mu_X)(Y -\mu_Y)\rangle}{\sigma_X\sigma_Y}$$
    where for the online algorithm the Welford algorithm taken from Donald E.
    Knuth were used [^2].

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
        online: bool = False,
        normalize_method: Optional[NormString] = None,
        knn_estimator: bool = False,
    ):
        """Initialize Similarity class."""
        self._metric: MetricString = metric
        self._online: bool = online
        self._knn_estimate: bool = knn_estimator
        if self._metric == 'NMI':
            if normalize_method is None:
                normalize_method = self._default_normalize_method
            self._normalize_method: NormString = normalize_method
        elif normalize_method is not None:
            raise NotImplementedError(
                'Normalize methods are only supported with metric="NMI"',
            )
        if self._metric != 'GY' and self._knn_estimate:
            raise NotImplementedError(
                (
                    'The mutual information estimate based on k-nearest'
                    'neighbors distances is only supported with metric="GY"'
                )
            )

    @singledispatchmethod
    @beartype
    def fit(
        self,
        X: Union[FloatMax2DArray, str],
        y: Optional[ArrayLikeFloat] = None,
    ) -> None:
        """Compute the correlation/nmi distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if online=True
            Training data.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        """
        raise NotImplementedError('Fatal error, this should never be reached.')

    @fit.register
    def _(self, X: np.ndarray, y=None) -> None:
        """Dispatched for online=False with matrix input."""
        self._reset()

        corr: np.ndarray
        matrix_: np.ndarray
        # parse data
        if self._online:
            raise TypeError('Using online=True requires X:str')
        if X.ndim == 1:
            raise ValueError(
                'Reshape your data either using array.reshape(-1, 1) if your '
                'data has a single feature or array.reshape(1, -1) if it '
                'contains a single sample.',
            )
        n_features: int
        n_samples: int
        n_samples, n_features = X.shape

        self._n_samples: int = n_samples
        self._n_features: int = n_features

        X: np.ndarray = _standard_scaler(X)
        if self._metric == 'correlation':
            corr = _correlation(X)
            matrix_ = np.abs(corr)
        elif self._metric == 'GY' and self._knn_estimate:
            matrix_ = self._nonlinear_GY_knn(X)
        else:  # 'NMI', 'JSD', 'GY
            matrix_ = self._nonlinear_correlation(X)
        self.matrix_: np.ndarray = np.clip(matrix_, a_min=0, a_max=1)

    @fit.register
    def _(self, X: str, y=None) -> None:
        """Dispatched for online=True with string."""
        self._reset()

        corr: np.ndarray
        matrix_: np.ndarray
        # parse data
        if not self._online:
            raise ValueError('Mode online=False reuqires X:np.ndarray.')

        if self._metric == 'correlation':
            corr = self._online_correlation(X)
            matrix_ = np.abs(corr)
        else:
            raise ValueError(
                'Mode online=True is only implemented for correlation.',
            )

        self.matrix_: np.ndarray = np.clip(matrix_, a_min=0, a_max=1)

    @beartype
    def _reset(self) -> None:
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, '_filename'):  # noqa: WPS421
            del self._filename  # noqa: WPS420
        if hasattr(self, 'matrix_'):  # noqa: WPS421
            del self.matrix_  # noqa: WPS420

    @beartype
    def _nonlinear_correlation(self, X: Float2DArray) -> FloatMatrix:
        """Return the nonlinear correlation."""
        calc_nl_corr: Callable = {
            'NMI': self._nmi,
            'GY': self._gy,
            'JSD': self._jsd,
        }[self._metric]

        nl_corr: FloatMatrix = np.empty(  # noqa: WPS317
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        for idx_i in range(self._n_features):
            xi = X[:, idx_i]
            nl_corr[idx_i, idx_i] = 1
            for idx_j in range(idx_i + 1, self._n_features):
                xj = X[:, idx_j]
                nl_corr_ij = calc_nl_corr(*_estimate_densities(xi, xj))
                nl_corr[idx_i, idx_j] = nl_corr_ij
                nl_corr[idx_j, idx_i] = nl_corr_ij

        return nl_corr

    @beartype
    def _nonlinear_GY_knn(self, X: Float2DArray) -> FloatMatrix:
        """Return the nonlinear correlation matrix based on Gel'fand-Yaglom
        based on a reliable knn-estimate of the mutual information."""
        nl_knn_corr = _knn_mutual_information(X, self._n_features)
        return np.sqrt(
            1 - np.exp(-2 * nl_knn_corr),
        )

    @beartype
    def _gy(
        self,
        pij: Float2DArray,
        pipj: Float2DArray,
        pi: Float1DArray,
        pj: Float1DArray,
    ) -> float:
        """Return the dissimilarity based on Gel'fand-Yaglom."""
        mutual_info: float = _kullback(pij, pipj)
        return np.sqrt(
            1 - np.exp(-2 * mutual_info),
        )

    @beartype
    def _jsd(
        self,
        pij: Float2DArray,
        pipj: Float2DArray,
        pi: Float1DArray,
        pj: Float1DArray,
    ) -> float:
        """Return the Jensen-Shannon based dissimilarity."""
        return jensenshannon(
            pij.flatten(),
            pipj.flatten(),
            base=2,
        )

    @beartype
    def _nmi(
        self,
        pij: Float2DArray,
        pipj: Float2DArray,
        pi: Float1DArray,
        pj: Float1DArray,
    ) -> float:
        """Return the Jensen-Shannon based dissimilarity."""
        mutual_info: float = _kullback(pij, pipj)
        normalization: float = self._normalization(pi, pj, pij)
        return mutual_info / normalization

    @beartype
    def _normalization(
        self, pi: np.ndarray, pj: np.ndarray, pij: np.ndarray,
    ) -> float:
        """Calculate the normalization factor for the MI matrix."""
        method: str = self._normalize_method
        if method == 'joint':
            return _entropy(pij)

        func: Callable = {
            'geometric': lambda arr: np.sqrt(np.prod(arr)),
            'arithmetic': np.mean,
            'min': np.min,
            'max': np.max,
        }[method]
        return func([_entropy(pi), _entropy(pj)])

    @beartype
    def _online_correlation(self, X: str) -> FloatMatrix:
        """Calculate correlation on the fly."""
        self._filename: str = X
        self._n_features: int = len(next(self._data_gen()))
        # parse mean, std and corr
        return self._welford_correlation()

    @beartype
    def _data_gen(
        self, comments: Union[str, Tuple[str, ...]] = ('#', '@'),
    ) -> Generator[Float1DArray, None, None]:
        """Return all non comment lines as generator."""
        with open(self._filename) as file_obj:
            for line in file_obj:
                if line.startswith(comments):
                    continue
                yield np.array(line.split()).astype(self._dtype)

    @beartype
    def _welford_correlation(self) -> FloatMatrix:
        """Calculate the correlation via online Welford algorithm.

        Welford algorithm, generalized to correlation. Taken from:
        Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
        Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

        """
        n: int = 0
        mean: np.ndarray = np.zeros(self._n_features, dtype=self._dtype)
        corr: np.ndarray = np.zeros(  # noqa: WPS317
            (self._n_features, self._n_features), dtype=self._dtype,
        )

        for x in self._data_gen():
            n += 1
            dx: np.ndarray = x - mean
            mean = mean + dx / n
            corr = corr + dx.reshape(-1, 1) * (
                x - mean
            ).reshape(1, -1)

        self._n_samples = n
        if n < 2:
            return np.full_like(corr, np.nan)

        std = np.sqrt(np.diag(corr) / (n - 1))
        return corr / (n - 1) / (
            std.reshape(-1, 1) * std.reshape(1, -1)
        )
