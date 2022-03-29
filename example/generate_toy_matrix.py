# -*- coding: utf-8 -*-
"""Script for generating toy model.

This script was used to generate the toy model of following publication.
It can be easily tweaked to generate a desired matrix, simply change the
global variables `NDIM` and `CLUSTS`.
> G. Diez, D. Nagel, and G. Stock,
> *Correlation-based feature selection to identify functional dynamcis
> in proteins*,
> in preparation

MIT License
Copyright (c) 2022, Daniel Nagel
All rights reserved.

"""
import numpy as np
from numpy.random import default_rng
from scipy.special import erf


# these parameters define the overall shape of the matrix
# one needs to fulfill, np.sum(CLUSTS) < NDIM
NDIM = 30  # dimension of the matrix
CLUSTS = np.array([10, 6, 4])  # list of cluster sizes


def noise(samples, width=1):
    """Generate noise coordinate."""
    width = np.random.choice(np.linspace(0.1, width, 3))**2
    return np.random.uniform(-0.5 * width, 0.5 * width, size=samples)


def make_toy_matrix():
    """Generate toy model coordinates."""
    # fix random seed to make dataset reproducible
    magic_number = 42
    rng = default_rng(seed=magic_number)

    mat = np.exp(
        rng.uniform(
            low=-NDIM, high=0, size=(NDIM, NDIM)
        )
    )
    mat = 0.5 * (mat + mat.T)

    for offset, dim in zip(np.cumsum([0, *CLUSTS[:-1]]), CLUSTS):
        clust = rng.uniform(low=0.9, high=1, size=(dim, dim))
        clust = clust * (
            erf(
                rng.uniform(low=0, high=2, size=dim)
            ) * erf(
                rng.uniform(low=0, high=2, size=dim)
            ).reshape(-1, 1)
        ) + mat[offset:offset + dim, offset:offset + dim]
        clust = 0.5 * (clust + clust.T)
        clust = clust[
            np.ix_(
                np.argsort(clust.sum(axis=0))[::-1],
                np.argsort(clust.sum(axis=1))[::-1],
            )
        ]
        mat[offset:offset + dim, offset:offset + dim] = clust

    mat = np.clip(mat, a_min=0, a_max=1)
    mat[np.diag_indices_from(mat)] = 1

    np.savetxt(
        'toy_correlation_matrix',
        mat,
        header='Toy matrix to compare different clustering approaches.',
        fmt='%.4f',
    )


if __name__ == '__main__':
    make_toy_matrix()
