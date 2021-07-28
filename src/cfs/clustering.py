# -*- coding: utf-8 -*-
"""Class for clustering the correlation matrices.

MIT License
Copyright (c) 2021, Daniel Nagel, Georg Diez
All rights reserved.

"""
import numpy as np
import igraph as ig
import leidenalg as la


class Clustering:
    """Class for clustering a correlation matrix."""

    def __init__(self, matrix):
        """Class for clustering a correlation matrix.

        Parameters
        ----------
        matrix : ndarray of shape(n_features, n_features)
            The linear/nonlinear correlation matrix which will be clustered.
        """
        self.matrix = matrix

