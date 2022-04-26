# -*- coding: utf-8 -*-
"""Class with helper functions.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
__all__ = ['load_clusters']  # noqa: WPS410

import datetime
import getpass
import platform
import sys

import numpy as np
from beartype import beartype

from mosaic._typing import (  # noqa: WPS436
    Object1DArray,
)


def get_rui(submod):
    """Get the runetime user information, to store as comment."""
    # get time without microseconds
    date = datetime.datetime.now()
    date = date.isoformat(sep=' ', timespec='seconds')

    rui = {
        'user': getpass.getuser(),
        'pc': platform.node(),
        'date': date,
        'args': ' '.join(sys.argv),
        'submod': '' if submod is None else f' {submod}',
    }

    return (
        'This file was generated by mosaic{submod}:\n{args}' +
        '\n\n{date}, {user}@{pc}'
    ).format(**rui)


def savetxt(filename, array, fmt, submodule=None, header=None):
    """Save ndarray with user runtime information."""
    header_generic = get_rui(submodule)
    if header:
        header_generic = f'{header_generic}\n\n{header}'

    np.savetxt(
        filename,
        array,
        fmt=fmt,
        header=header_generic,
    )


@beartype
def load_clusters(filename: str) -> Object1DArray:
    """Load clusters stored from cli.

    Parameters
    ----------
    filename : str
        Filename of cluster file.

    """
    clusters_list = [
        np.array(
            cluster.split()
        ).astype(int).tolist()
        for cluster in np.loadtxt(filename, delimiter='\n', dtype=str)
    ]

    # In case of clusters of same length, numpy casted it as a 2D array.
    # To ensure that the result is an numpy array of list, we need to
    # create an empty list, adding the values in the second step
    clusters: Object1DArray = np.empty(len(clusters_list), dtype=object)
    clusters[:] = clusters_list
    return clusters


@beartype
def save_clusters(filename: str, clusters: Object1DArray):
    clusters_string = np.array(
        [
            ' '.join([str(state) for state in cluster])
            for cluster in clusters
        ],
        dtype=str,
    )
    savetxt(
        filename,
        clusters_string,
        fmt='%s',
        submodule='clustering',
        header=(
            'In ith row are the indices listed (zero-indexed) corresponding '
            'to cluster i.'
        ),
    )
