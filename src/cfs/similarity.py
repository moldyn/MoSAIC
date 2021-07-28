# -*- coding: utf-8 -*-
"""Class for estimating correlation matrix.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import jensenshannon


class Similarity:  # noqa: WPS214
    """Class for calculating similarity measure.

    Parameters
    ----------
    metric : str, default='correlation'
        the correlation metric to use for the distance matrix.

        - 'correlation' will use absolute value of the Pearson correlation
        - 'NMI' will use mutual information normalized by joined entropy
        - 'JSD' will use the Jensen-Shannon divergence between the joint
          probability distribution and the product of the marginal probability
          distributions to calculate their dissimilarity

        Note: 'NMI' is supported only with online=False

    online : bool, default=False
        If True the input of fit X needs to be a file name and the correlation
        is calculated on the fly. Otherwise, an array is assumed as input X.

    normalize_method : str, default='individual'
        Only necessary for metric 'nmi'. Determines the normalization factor
        for the mutual information.

        - 'individual' is the square root of the product of the individual
          entropies
        - 'joint' is the joint entropy

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
    $$\rho_{X,Y} = \frac{\langle(X -\mu_X)(Y -\mu_Y)\rangle}{\sigma_X\sigma_Y}$$
    where for the online algorithm the Welford algorithm taken from Donald E.
    Knuth were used [1][Knuth98].

    [Knuth98]: Welford algorithm, generalized to correlation. Taken from:
    Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
    Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

    The Jensen-Shannon divergence is defined as
    $$D_{\text{JS}} = \frac{1}{2} D_{\text{KL}}(p(x,y)||M) + \frac{1}{2} D_{\text{KL}}(p(x)p(y)||M)$$,
    where $M = \frac{1}{2} [p(x,y) + p(x)p(y)]$ is an averaged probability
    distribution and $D_{\text{KL}}$ denotes the Kullback-Leibler divergence.

    """
    _dtype = np.float64
    _default_normalize_method = 'individual'

    def __init__(
        self,
        *,
        metric='correlation',
        online=False,
        normalize_method=None,
    ):
        """Initialize Similarity class."""
        self._metric = metric
        self._online = online
        if self._metric == 'NMI':
            if normalize_method is None:
                normalize_method = self._default_normalize_method
            self._normalize_method = normalize_method
        elif self._normalize_method:
            raise NotImplementedError(
                'Normalize methods are only supported with metric="NMI"'
            )

    def _reset(self):
        """Reset internal data-dependent state of correlation."""
        if hasattr(self, '_is_file'):
            del self._is_file
            del self.matrix_

    def fit(self, X, y=None):
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

        # parse data
        if self._online:
            if self._metric == 'correlation':
                corr = self._online_correlation(X)
                matrix_ = np.abs(corr)
            else:
                raise ValueError(
                    'Mode online=True is only implemented for correlation.'
                )
        else:
            self._n_samples, self._n_features = X.shape
            X = self._standard_scaler(X)
            if self._metric == 'correlation':
                corr = self._correlation(X)
                matrix_ = np.abs(corr)
            elif self._metric == 'nmi':
                nmi = self._nmi(X)
                matrix_ = nmi
            elif self._metric == 'jsd':
                jsd = self._jsd(X)
                matrix_ = 1 - jsd

        self.matrix_ = np.clip(matrix_, a_min=0, a_max=1)

    def _correlation(self, X):
        """Return the correlation."""
        corr = np.empty(
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        for idx_i in range(self._n_features):
            xi = X[:, idx_i]
            corr[idx_i, idx_i] = 1
            for idx_j in range(idx_i + 1, self._n_features):
                xj = X[:, idx_j]
                correlation = np.dot(xi, xj) / self._n_samples
                corr[idx_i, idx_j] = corr[idx_j, idx_i] = correlation
        return corr

    def _nmi(self, X):
        """Returns the normalized mutual information matrix."""
        X = self._standard_scaler(X)
        nmi = np.empty(
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        for idx_i in range(self._n_features):
            xi = X[:, idx_i]
            nmi[idx_i, idx_i] = 1
            for idx_j in range(idx_i + 1, self._n_features):
                xj = X[:, idx_j]
                # Calculate joint and marginal probability density
                p_ij = self._estimate_density(xi, xj)
                p_i = np.sum(p_ij, axis=1)
                p_j = np.sum(p_ij, axis=0)
                pi_times_pj = p_i[:, np.newaxis] * p_j[np.newaxis, :]
                mutual_info = np.sum(
                    p_ij * np.ma.log(np.ma.divide(p_ij, pi_times_pj))
                )
                normalization = self._normalization(p_i, p_j, p_ij)
                normalized_mi = mutual_info / normalization
                nmi[idx_i, idx_j] = nmi[idx_j, idx_i] = normalized_mi

        return nmi

    def _jsd(self, X):
        """Returns the Jensen-Shannon based dissimilarity"""
        X = self._standard_scaler(X)
        jsd = np.empty(
            (self._n_features, self._n_features), dtype=self._dtype
        )
        for idx_i in range(self._n_features):
            xi = X[:, idx_i]
            jsd[idx_i, idx_i] = 0
            for idx_j in range(idx_j + 1, self._n_features):
                xj = X[:, idx_j]
                p_ij = self._estimate_density(xi, xj)
                p_i = np.sum(p_ij, axis=1)
                p_j = np.sum(p_ij, axis=0)
                pi_times_pj = p_i[:, np.newaxis] * p_j[np.newaxis, :]
                jsdivergence = jensenshannon(
                    p_ij.flatten(),
                    pi_times_pj.flatten(),
                    base=2,
                )
                # 1-jsd to get similarity instead of dissimilarity
                jsd[idx_i, idx_j] = jsd[idx_j, idx_i] = jsdivergence

        return jsd

    def _normalization(self, p_i, p_j, p_ij):
        """Calculates the normalization factor for the MI matrix."""
        method = self._normalize_method
        if method == 'individual':
            return np.sqrt(
                np.sum(p_i * np.ma.log(p_i) * np.sum(p_j * np.ma.log(p_j)))
            )
        if method == 'joint':
            return np.sum(p_ij * np.ma.log(p_ij))

    def _online_correlation(self, X):
        """Calculate correlation on the fly."""
        self._filename = X
        self._n_features = len(next(self._data_gen()))
        # parse mean, std and corr
        return self._welford_correlation()

    def _data_gen(self, comments=('#', '@')):
        """Generator for looping over file."""
        for line in open(self._filename):
            if line.startswith(comments):
                continue
            yield np.array(line.split()).astype(self._dtype)

    def _welford_correlation(self):
        """Calculate the correlation via online Welford algorithm.

        Welford algorithm, generalized to correlation. Taken from:
        Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
        Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

        """
        n = 0
        mean = np.zeros(self._n_features, dtype=self._dtype)
        corr = np.zeros((self._n_features, self._n_features), dtype=self._dtype)

        for x in self._data_gen():
            n += 1
            dx = x - mean
            mean = mean + dx / n
            corr = corr + dx.reshape(-1, 1) * (x - mean).reshape(1, -1)

        self._n_samples = n
        if n < 2:
            return np.full_like(corr, np.nan)
        else:
            std = np.sqrt(np.diag(corr) / (n - 1))
            return corr / (n - 1) / (
                std.reshape(-1, 1) * std.reshape(1, -1)
            )

    def _check_input_with_params(self, X):
        # check if is string
        is_file = isinstance(X, str)
        if is_file and not self._online:
            raise TypeError(
                'Filename input is supported only with online=True'
            )

        is_array = isinstance(X, np.ndarray)
        error_dim1 = (
            'Reshape your data either using array.reshape(-1, 1) if your data '
            'has a single feature or array.reshape(1, -1) if it contains a '
            'single sample.'
        )
        if is_array:
            if X.ndim == 1:
                raise ValueError(error_dim1)
            elif X.ndim > 2:
                raise ValueError(
                    f'Found array with dim {X.ndim} but dim=2 expected.'
                )

        if not is_file and not is_array:
            if self._online:
                raise TypeError(
                    'Input needs to be of type "str" (filename), but '
                    f'"{type(X)}" given'
                )
            else:
                raise TypeError(
                    'Input needs to be of type "ndarray", but '
                    f'"{type(X)}" given'
                )

    @staticmethod
    def _standard_scaler(X):
        """Make data mean-free and std=1."""
        scaler = preprocessing.StandardScaler().fit(X)
        return scaler.transform(X)

    @staticmethod
    def _estimate_density(x, y, bins=100):
        """Calculates two dimensional probability density."""
        hist, _, _ = np.histogram2d(x, y, bins, density=True)
        # transpose since numpy considers axis 0 as y and axis 1 as x
        hist_transposed = hist.T

        return hist_transposed / np.sum(hist)
