# -*- coding: utf-8 -*-
"""Class for defining typing.

In this module all type hint types are defined. This includes as well the
available metrics, norms and clustering methods

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import numpy as np
from beartype.typing import List, Union
from beartype.vale import Is, IsAttr, IsEqual
from mosaic import METRICS, MODES, NORMS

try:  # for python <= 3.8 use typing_extensions
    from beartype.typing import Annotated
except ImportError:
    from typing_extensions import Annotated


def _get_resolution(x):
    dtype = np.result_type(x)
    if np.issubdtype(dtype, np.integer):
        return 0
    return np.finfo(dtype).resolution


def _allclose(x, y) -> bool:
    """Wrapper around np.allclose with dtype dependent atol."""
    atol = np.max([
        _get_resolution(x),
        _get_resolution(y),
        # default value of numpy
        1e-8,
    ])
    return np.allclose(x, y, atol=atol)


class NDim:
    """Class for creating Validators checking for desired dimensions."""
    def __class_getitem__(self, ndim):
        return IsAttr['ndim', IsEqual[ndim]]


class DType:
    """Class for creating Validators checking for desired dtype."""
    def __class_getitem__(self, dtype):
        return Is[lambda arr: np.issubdtype(arr.dtype, dtype)]


# Define Validators
# cast np.bool_ return type of np.all to bool to avoid tri-state boolean
# error, see beartype #153
IsDiagonalOne = Is[lambda arr: _allclose(np.diag(arr), 1)]
IsDTypeLike = Is[lambda dtype: np.issubdtype(dtype, np.generic)]
IsLessThanOne = Is[lambda arr: bool(np.all(arr <= 1))]
IsMatrix = Is[lambda arr: arr.shape[0] == arr.shape[1]]
IsMetricString = Is[lambda val: val in METRICS]
IsModeString = Is[lambda val: val in MODES]
IsNormString = Is[lambda val: val in NORMS]
IsPositive = Is[lambda arr: bool(np.all(arr >= 0))]
IsStrictlyPositive = Is[lambda arr: bool(np.all(arr > 0))]
IsSymmetric = Is[lambda arr: _allclose(arr, arr.T)]

# Define Types
# String (enum-type) datatypes
MetricString = Annotated[str, IsMetricString]
NormString = Annotated[str, IsNormString]
ClusteringModeString = Annotated[str, IsModeString]

# scalar datatypes
PositiveInt = Annotated[
    Union[int, np.integer],
    IsStrictlyPositive,
]
Int = Union[int, np.integer]
Float = Union[float, np.floating]
NumInRange0to1 = Annotated[
    Union[int, float, np.integer, np.floating],
    IsPositive & IsLessThanOne,
]

# beartype substitute for np.typing.DTypeLike
DTypeLike = Annotated[type, IsDTypeLike]

# array datatypes
FloatNDArray = Annotated[np.ndarray, DType[np.floating]]
IntNDArray = Annotated[np.ndarray, DType[np.integer]]
ObjectNDArray = Annotated[np.ndarray, DType[object]]
ArrayLikeFloat = Union[List[float], FloatNDArray]
Index1DArray = Annotated[IntNDArray, NDim[1] & IsPositive]
Float1DArray = Annotated[FloatNDArray, NDim[1]]
Float2DArray = Annotated[FloatNDArray, NDim[2]]
FloatMatrix = Annotated[Float2DArray, IsMatrix]
SimilarityMatrix = Annotated[
    FloatMatrix,
    IsPositive & IsLessThanOne & IsSymmetric & IsDiagonalOne,
]
FloatMax2DArray = Annotated[FloatNDArray, NDim[1] | NDim[2]]
Object1DArray = Annotated[ObjectNDArray, NDim[1]]
ObjectMax2DArray = Annotated[ObjectNDArray, NDim[1] | NDim[2]]
