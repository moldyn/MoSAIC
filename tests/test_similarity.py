# -*- coding: utf-8 -*-
"""Tests for the similarity measure.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import os.path

from beartype.roar import BeartypeException
import numpy as np
import pytest

import cfs

# Current directory
HERE = os.path.dirname(__file__)


@pytest.fixture
def coords_file():
    """Define coordinate file."""
    return os.path.join(HERE, 'amino_120d_200000')


@pytest.mark.parametrize('density, entropy, error', [
    ([1, np.e, 1], -np.e, None),
    (np.arange(10), -79.05697962199447, None),
    (['a', 'b'], None, BeartypeException),
    (15, None, BeartypeException),
])
def test__entropy(density, entropy, error):
    if not error:
        np.testing.assert_almost_equal(
            cfs.similarity._entropy(density), entropy,
        )
    else:
        with pytest.raises(error):
            cfs.similarity._entropy(density)