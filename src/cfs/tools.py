# -*- coding: utf-8 -*-
"""Set of helpful functions.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import numba
import numpy as np

def dummy_func(x, y, bins=250):
    """Shift integer array (data) from old to new values.

    > **CAUTION:**
    > The values of `val_old`, `val_new` and `data` needs to be integers.

    The basic function is based on Ashwini_Chaudhary solution:
    https://stackoverflow.com/a/29408060

    Parameters
    ----------
    array : StateTraj or ndarray or list or list of ndarrays
        1D data or a list of data.

    val_old : ndarray or list
        Values in data which should be replaced. All values needs to be within
        the range of `[data.min(), data.max()]`

    val_new : ndarray or list
        Values which will be used instead of old ones.

    dtype : data-type, optional
        The desired data-type. Needs to be of type unsigned integer.

    Returns
    -------
    array : ndarray
        Shifted data in same shape as input.

    """
    pass