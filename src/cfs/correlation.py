# -*- coding: utf-8 -*-
"""Class for estimating correlation matrix.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
from enum import Enum, auto

import numpy as np
from sklearn import preprocessing
    

class Correlation:  # noqa: WPS214
    """Class for handling input data.

    Parameters
    ----------
    metric : str, default='correlation'
        the correlation metric to use for the distance matrix.

        - 'correlation' will use absolute value of the Pearson correlation
        - 'nmi' will use mutual information normalized by joined entropy

        Note: 'nmi' is supported only 
    
    online : bool, default=True
        If True the input of fit X needs to be a file name and the correlation
        is calculated on the fly. Otherwise, an array is assumed as input X.
    
    """

    def __init__(self, *, metric='correlation', online=True):
        """Initialize Correlation class."""
        self._metric = metric
        self._online = online
    
    def _reset(self):
        """Reset internal data-dependent state of correlation."""
        pass
        #if hasattr(self, 'a')
        self._mean = self._std = self._corr = None
    
    def fit(self, X, y=None):
        """Compute the correlation/nmi distance matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or str if online=True
            Training data.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        """
        self._parse_input(X)
        self._dtype = np.float64

        # reset values on changeing input source

        if self._is_file:
            self._filename = X
            self._n_features = len(next(self._data_gen()))
            # parse mean, std and corr
            self._welford()
        else:
            self._n_samples, self._n_features = input.shape
            self._dtype = input.dtype

            scaler = preprocessing.StandardScaler().fit(X)
            self._data = scaler.transform(X)

    def correlation(self):
        """Return the correlation.

        Calculate the correlation matrix
        $$\rho_{X,Y} = \frac{\langle(X -\mu_X)(Y -\mu_Y)\rangle}{\sigma_X\sigma_Y}$$

        Returns
        ----------
        corr : ndarray of shape (n_features, n_features)
            Correlation matrix bound to [-1, 1].

        """

        if self._is_file:
            corr = self._corr
        else:
            corr = np.empty(
                (self._n_features, self._n_features), dtype=self._dtype,
            )
            for i in range(self._n_features):
                xi = self._data[:, i]
                corr[i, i] = 1
                for j in range(i + 1,self._n_features):
                    xj = self._data[:, i]
                    corr[i, j] = corr[j, i] = np.dot(xi, xj) / self._n_samples

        return np.clip(corr, a_min=-1, a_max=1)

    def _data_gen(
        self,
        comments=('#', '@'),
        scaler=False,
    ):
        if scaler:
            if self._mean is None or self._std is None:
                raise ValueError('First run needs to be with scaler=False')

        for line in open(self._filename):
            if line.startswith(comments):
                continue
            if scaler:
                yield (
                    np.array(line.split()).astype(self._dtype) - self._mean
                ) / self._std
            else:
                yield np.array(line.split()).astype(self._dtype)
    
    def _welford(self):
        """Calculate the online welford mean and variance.

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
        self._mean = mean
        if n < 2:
            self._std = np.full_like(mean, np.nan)
            self._corr = np.full_like(corr, np.nan)
        else:
            self._std = np.sqrt(np.diag(corr) / (n - 1))
            self._corr = corr / (n - 1) / (self._std.reshape(-1, 1) * self._std.reshape(1, -1))

    def _parse_input(self, data):
        # check if is string
        self._is_file = isinstance(data, str)
        if self._is_file and not self._online:
            raise ValueError(
                'Filename input is supported only with online=True'
            )

        self._is_array = isinstance(data, np.ndarray)
        error_dim1 = (
            'Reshape your data either using array.reshape(-1, 1) if your data '
            'has a single feature or array.reshape(1, -1) if it contains a '
            'single sample.'
        ) 
        if self._is_array:
            if data.ndim == 1:
                raise ValueError(error_dim1)
            elif data.ndim > 2:
                raise ValueError(
                    f'Found array with dim {data.ndim} but dim=2 expected.'
                )
            
        if not self._is_file and not self._is_array:
            if self._online:
                raise ValueError(
                    'Input needs to be of type "str" (filename), but '
                    f'"{type(data)}" given'
                )
            else:
        
