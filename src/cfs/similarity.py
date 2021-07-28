# -*- coding: utf-8 -*-
"""Class for estimating correlation matrix.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
import numpy as np
from sklearn import preprocessing


class Similarity:  # noqa: WPS214
    """Class for calculating similarity measure.

    Parameters
    ----------
    metric : str, default='correlation'
        the correlation metric to use for the distance matrix.

        - 'correlation' will use absolute value of the Pearson correlation
        - 'nmi' will use mutual information normalized by joined entropy

        Note: 'nmi' is supported only with online=False

    online : bool, default=False
        If True the input of fit X needs to be a file name and the correlation
        is calculated on the fly. Otherwise, an array is assumed as input X.

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
    >>> diss = Dissimilarity()
    >>> diss.fit(data)
    >>> diss.matrix_
    array([[1.        , 0.91666054],
           [0.91666054, 1.        ]])

    Notes
    -----

    The correlation is defined by
    $$\rho_{X,Y} = \frac{\langle(X -\mu_X)(Y -\mu_Y)\rangle}{\sigma_X\sigma_Y}$$
    where for the online algorithm the Welford algorithm taken from Donald E.
    Knuth were used [1][Knuth98].

    [Knuth98]: Welford algorithm, generalized to correlation. Taken from:
    Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
    Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

    """
    _dtype = np.float64

    def __init__(self, *, metric='correlation', online=False):
        """Initialize Similarity class."""
        self._metric = metric
        self._online = online

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
        self._is_file = self._is_file_input(X)

        # parse data
        if self._is_file:
            if self._metric == 'correlation':
                self._filename = X
                self._n_features = len(next(self._data_gen()))
                # parse mean, std and corr
                corr = self._welford()
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
                matrix_ = 1 - nmi

        self.matrix_ = np.clip(matrix_, a_min=0, a_max=1)

    def _correlation(self, X):
        """Return the correlation."""
        corr = np.empty(
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        for i in range(self._n_features):
            xi = X[:, i]
            corr[i, i] = 1
            for j in range(i + 1, self._n_features):
                xj = X[:, j]
                corr[i, j] = corr[j, i] = np.dot(xi, xj) / self._n_samples
        return corr

    def _mi(self, X):
        """Returns the mutual information matrix."""
        X = self._standard_scaler(X)
        mi = np.empty(
            (self._n_features, self._n_features), dtype=self._dtype,
        )
        for i in range(self._n_features):
            xi = X[:, i]
            mi[i, i] = 1
            for j in range(i + 1, self._n_features):
                xj = X[:, j]
                # Calculate joint and marginal probability density
                p_ij = self._estimate_density(xi, xj)
                p_i = np.sum(p_ij, axis=1)
                p_j = np.sum(p_ij, axis=0)
                pi_times_pj = p_i[:, np.newaxis] * p_j[np.newaxis, :]
                mutual_info = np.sum(
                    p_ij * np.ma.log(np.ma.divide(p_ij, pi_times_pj))
                )
                mi[i, j] = mi[j, i] = mutual_info

        return mi

    def _nmi(self, ):
        """Returns the normalized mutual information matrix."""

    def _data_gen(self, comments=('#', '@')):
        """Generator for looping over file."""
        for line in open(self._filename):
            if line.startswith(comments):
                continue
            yield np.array(line.split()).astype(self._dtype)

    def _welford(self):
        """Calculate the online welford mean, variance and correlation.

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

    def _is_file_input(self, X):
        # check if is string
        is_file = isinstance(X, str)
        if is_file and not self._online:
            raise ValueError(
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
                raise ValueError(
                    'Input needs to be of type "str" (filename), but '
                    f'"{type(X)}" given'
                )
            else:
                raise ValueError(
                    'Input needs to be of type "ndarray", but '
                    f'"{type(X)}" given'
                )

        return is_file

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
