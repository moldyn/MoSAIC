# -*- coding: utf-8 -*-
"""Class for handling input data.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import numpy as np
from sklearn import preprocessing


class StandardScaler:  # noqa: WPS214
    """Class for handling input data."""

    def __init__(self, data=None):
        """Initialize StandardScalar.

        If called with string (filename), online methods will be used which is
        much slower but needs less memory.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features) or str
            The data used to create scaler or a path to the file to be parsed.

        """
        self.data = data

    @property
    def data(self):
        """Return normalized data with zero mean and std=1.

        Returns
        -------
        data : ndarray of shape (n_samples, n_features) or generator
            Array holding normalized data or generator iterating over file.

        """
        if self._is_file:
            return self._data_gen(scaler=True)
        else:
            return self._data

    @data.setter
    def data(self, data):
        """Set the input data.

        Parameters
        ----------
        trajs : list of ndarrays
            List of ndarrays holding the input data.

        """
        self._parse_input(data)
        self._dtype = np.float64

        # reset values on changeing input source
        self._mean = self._std = self._corr = None

        if self._is_file:
            self._filename = data
            self._n_features = len(next(self._data_gen()))
            # parse mean, std and corr
            self._welford()
        else:
            self._n_samples, self._n_features = input.shape
            self._dtype = input.dtype

            scaler = preprocessing.StandardScaler().fit(data)
            self._data = scaler.transform(data)

    def __repr__(self):
        """Return representation of class."""
        kw = {
            'clname': self.__class__.__name__,
            'filename': self._filename,
            'data': self._data,
        }
        if self._is_file:
            return ('{clname}({filename})'.format(**kw))
        else:
            return ('{clname}({data})'.format(**kw))

    def __str__(self):
        """Return string representation of class."""
        kw = {
            'clname': self.__class__.__name__,
            'filename': self._filename,
            'data': self._data,
        }
        if self._is_file:
            return ('{clname}({filename!s})'.format(**kw))
        else:
            return ('{clname}({data!s})'.format(**kw))

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
            raise ValueError(
                'StandardScaler can be constructed only from "str" or '
                f'"np.ndarray", but "{type(data)}" given'
            )