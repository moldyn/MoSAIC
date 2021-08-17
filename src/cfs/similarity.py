# -*- coding: utf-8 -*-
"""Class for estimating correlation matrix.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
from beartype import beartype
from scipy.spatial.distance import jensenshannon
from sklearn import preprocessing


@beartype
def _entropy(p: np.ndarray) -> float:
    """Calculate entropy of density p."""
    return -1 * np.sum(p * np.ma.log(p))


@beartype
def _kullback(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Kullback-Leibler divergence of density p, q."""
    return np.sum(
        p * np.ma.log(np.ma.divide(p, q)),
    )


def _standard_scaler(X):
    """Make data mean-free and std=1."""
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)


@beartype
def _estimate_densities(
    x: np.ndarray, y: np.ndarray, bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate two dimensional probability densities."""
    hist, _, _ = np.histogram2d(x, y, bins, density=True)
    # transpose since numpy considers axis 0 as y and axis 1 as x
    pij = hist.T / np.sum(hist)
    pi = np.sum(pij, axis=1)
    pj = np.sum(pij, axis=0)
    pipj = pi[:, np.newaxis] * pj[np.newaxis, :]

    return pij, pipj, pi, pj


@beartype
def _correlation(X: np.ndarray) -> np.ndarray:
    """Return the correlation of input.

    Each feature (column) of X need to be mean-free with standard deviation 1.

    """
    return X.T / len(X) @ X


class Similarity:  # noqa: WPS214
    r"""Class for calculating the similarity measure.

    Parameters
    ----------
    metric : str, default='correlation'
        the correlation metric to use for the feature distance matrix.

        - 'correlation' will use the absolute value of the Pearson correlation
        - 'NMI' will use the mutual information normalized by joined entropy
        - 'GY' use Gel'fand and Yaglom normalization[^1]
        - 'JSD' will use the Jensen-Shannon divergence between the joint
          probability distribution and the product of the marginal probability
          distributions to calculate their dissimilarity

        Note: 'NMI' is supported only with online=False

    online : bool, default=False
        If True, the input of fit X needs to be a file name and the correlation
        is calculated on the fly. Otherwise, an array is assumed as input X.

    normalize_method : str, default='arithmetic'
        Only required for metric 'NMI'. Determines the normalization factor
        for the mutual information:

        - 'joint' is the joint entropy
        - 'max' is the maximum of the individual entropies
        - 'arithmetic' is the mean of the individual entropies
        - 'geometric' is the square root of the product of the individual
          entropies
        - 'min' is the minimum of the individual entropies

    Attributes
    ----------
    matrix_ : ndarray of shape (n_features, n_features)
        The correlation-measure-based pairwise distance matrix of the data. It
        scales from [0, 1].

    Examples
    --------
    >>> import cfs
    >>> x = np.linspace(0, np.pi, 1000)
    >>> data = np.array([np.cos(x), np.sin(x)]).T
    >>> sim = cfs.Similarity()
    >>> sim.fit(data)
    >>> sim.matrix_
    array([[1.        , 0.91666054],
           [0.91666054, 1.        ]])

    Notes
    -----
    The correlation is defined as
    $$\rho_{X,Y} =
    \frac{\langle(X -\mu_X)(Y -\mu_Y)\rangle}{\sigma_X\sigma_Y}$$
    where for the online algorithm the Welford algorithm taken from Donald E.
    Knuth were used [^2].

    [^1]: Gel'fand, I.M. and Yaglom, A.M. (1957). "Calculation of amount of
        information about a random function contained in another such function".
        American Mathematical Society Translations, series 2, 12, pp. 199–246.

    [^2]: Welford algorithm, generalized to correlation. Taken from:
        Donald E. Knuth (1998). "The Art of Computer Programming", volume 2:
        Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

    The Jensen-Shannon divergence is defined as
    $$D_{\text{JS}} = \frac{1}{2} D_{\text{KL}}(p(x,y)||M)
    + \frac{1}{2} D_{\text{KL}}(p(x)p(y)||M)\;,$$
    where \(M = \frac{1}{2} [p(x,y) + p(x)p(y)]\) is an averaged probability
    distribution and \(D_{\text{KL}}\) denotes the Kullback-Leibler divergence.

    """

    _dtype: np.dtype = np.float128
    _default_normalize_method: str = 'arithmetic'
    _available_metrics: Tuple[str, ...] = ('correlation', 'NMI', 'JSD', 'GY')
    _available_norms: Tuple[str, ...] = (
        'joint', 'geometric', 'arithmetic', 'min', 'max',
    )

    @beartype
    def __init__(
        self,
        *,
        metric: str = 'correlation',
        online: bool = False,
        normalize_method: Optional[str] = None,
    ):
        """Initialize Similarity class."""
        if metric not in self._available_metrics:
            metrics = ', '.join([f'"{m}"' for m in self._available_metrics])
            raise NotImplementedError(
                f'Metric {metric} is not implemented, use one of [{metrics}]',
            )

        self._metric: str = metric
        self._online: bool = online
        if self._metric == 'NMI':
            if normalize_method is None:
                normalize_method = self._default_normalize_method
            if normalize_method not in self._available_norms:
                norms = ', '.join([f'"{n}"' for n in self._available_norms])
                raise NotImplementedError(
                    f'Normalization method {normalize_method} is not '
                    f'implemented, use one of [{norms}]',
                )
            self._normalize_method: str = normalize_method
        elif normalize_method is not None:
            raise NotImplementedError(
                'Normalize methods are only supported with metric="NMI"',
            )

    @beartype
    def fit(self, X: np.ndarray, y=None) -> None:
        """Compute the correlation/nmi distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if online=True
            Training data.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        """
        self._reset()
        self._check_input_with_params(X)

        corr: np.ndarray
        matrix_: np.ndarray

        # parse data
        if self._online:
            if self._metric == 'correlation':
                corr = self._online_correlation(X)
                matrix_ = np.abs(corr)
            else:
                raise ValueError(
                    'Mode online=True is only implemented for correlation.',
                )
        else:
            n_features: int
            n_samples: int
            n_samples, n_features = X.shape

            self._n_samples: int = n_samples
            self._n_features: int = n_features

            X: np.ndarray = _standard_scaler(X)
            if self._metric == 'correlation':
                corr = _correlation(X)
                matrix_ = np.abs(corr)
            elif self._metric in {'NMI', 'JSD', 'GY'}:
                matrix_ = self._nonlinear_correlation(X)
            else:
                raise NotImplementedError(
                    f'Metric {self._metric} is not implemented',
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
    def _nonlinear_correlation(self, X: np.ndarray) -> np.ndarray:
        """Return the nonlinear correlation."""
        calc_nl_corr: Callable
        if self._metric == 'NMI':
            calc_nl_corr = self._nmi
        elif self._metric == 'GY':
            calc_nl_corr = self._gy
        else:
            calc_nl_corr = self._jsd

        nl_corr: np.ndarray = np.empty(  # noqa: WPS317
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
    def _gy(
        self,
        pij: np.ndarray,
        pipj: np.ndarray,
        pi: np.ndarray,
        pj: np.ndarray,
    ) -> float:
        """Return the Jensen-Shannon based dissimilarity."""
        mutual_info: float = _kullback(pij, pipj)
        return -0.5 * np.sqrt(
            1 - np.exp(-2 * mutual_info),
        )

    @beartype
    def _nmi(
        self,
        pij: np.ndarray,
        pipj: np.ndarray,
        pi: np.ndarray,
        pj: np.ndarray,
    ) -> float:
        """Return the Jensen-Shannon based dissimilarity."""
        mutual_info: float = _kullback(pij, pipj)
        normalization: float = self._normalization(pi, pj, pij)
        return mutual_info / normalization

    @beartype
    def _jsd(
        self,
        pij: np.ndarray,
        pipj: np.ndarray,
        pi: np.ndarray,
        pj: np.ndarray,
    ) -> float:
        """Return the Jensen-Shannon based dissimilarity."""
        return jensenshannon(
            pij.flatten(),
            pipj.flatten(),
            base=2,
        )

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
    def _online_correlation(self, X: str) -> np.ndarray:
        """Calculate correlation on the fly."""
        self._filename: str = X
        self._n_features: int = len(next(self._data_gen()))
        # parse mean, std and corr
        return self._welford_correlation()

    @beartype
    def _data_gen(
        self, comments: str = ('#', '@'),
    ) -> Generator[np.ndarray, None, None]:
        """Return all non comment lines as generator."""
        with open(self._filename) as file_obj:
            for line in file_obj:
                if line.startswith(comments):
                    continue
                yield np.array(line.split()).astype(self._dtype)

    @beartype
    def _welford_correlation(self) -> np.ndarray:
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

    @beartype
    def _check_input_with_params(self, X: Union[str, np.ndarray]) -> None:
        # check if is string
        is_file = isinstance(X, str)
        if is_file and not self._online:
            raise TypeError(
                'Filename input is supported only with online=True',
            )

        is_array = isinstance(X, np.ndarray)
        error_msg = None
        error_dim1 = (
            'Reshape your data either using array.reshape(-1, 1) if your data '
            'has a single feature or array.reshape(1, -1) if it contains a '
            'single sample.'
        )
        if is_array:
            if X.ndim == 1:
                error_msg = error_dim1
            elif X.ndim > 2:
                error_msg = (
                    f'Found array with dim {X.ndim} but dim=2 expected.'
                )

        if not is_file and not is_array:
            if self._online:
                error_msg = (
                    'Input needs to be of type "str" (filename), but '
                    f'"{type(X)}" given',
                )
            error_msg = (
                'Input needs to be of type "ndarray", but '
                f'"{type(X)}" given',
            )

        if error_msg:
            raise ValueError(error_msg)
