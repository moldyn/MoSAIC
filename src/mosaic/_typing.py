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
from beartype.vale import Is

try:  # for python <= 3.8 use typing_extensions
    from beartype.typing import Annotated
except ImportError:
    from typing_extensions import Annotated

# String (enum-type) datatypes
MetricString = Annotated[
    str, Is[lambda val: val in {'correlation', 'NMI', 'JSD', 'GY'}],
]
NormString = Annotated[
    str,
    Is[lambda val: val in {
        'joint', 'geometric', 'arithmetic', 'min', 'max',
    }],
]
ClusteringModeString = Annotated[
    str, Is[lambda val: val in {'CPM', 'modularity', 'linkage'}],
]

# scalar datatypes
PositiveInt = Annotated[int, Is[lambda val: val > 0]]
NumInRange0to1 = Annotated[
    Union[int, float, np.integer, np.floating], Is[lambda val: 0 <= val <= 1],
]

# array datatypes
FloatNDArray = Annotated[
    np.ndarray, Is[lambda arr: np.issubdtype(arr.dtype, np.floating)],
]
IntNDArray = Annotated[
    np.ndarray, Is[lambda arr: np.issubdtype(arr.dtype, np.integer)],
]
ObjectNDArray = Annotated[
    np.ndarray, Is[lambda arr: np.issubdtype(arr.dtype, object)],
]
ArrayLikeFloat = Union[List[float], FloatNDArray]
Index1DArray = Annotated[
    IntNDArray, Is[
        lambda arr: arr.ndim == 1 and np.all(arr >= 0)
    ],
]
Float1DArray = Annotated[
    FloatNDArray, Is[lambda arr: arr.ndim == 1],
]
Float2DArray = Annotated[
    FloatNDArray, Is[lambda arr: arr.ndim == 2],
]
FloatMatrix = Annotated[
    Float2DArray,
    Is[lambda arr: arr.shape[0] == arr.shape[1]],
]
SimilarityMatrix = Annotated[
    FloatMatrix,
    Is[
        lambda arr: (
            np.allclose(arr, arr.T) and
            np.allclose(np.diag(arr), 1) and
            np.all(arr <= 1) and
            np.all(arr >= 0)
        )
    ],
]
FloatMax2DArray = Annotated[
    FloatNDArray, Is[lambda arr: 1 <= arr.ndim <= 2],
]
Object1DArray = Annotated[
    ObjectNDArray, Is[lambda arr: arr.ndim == 1],
]
ObjectMax2DArray = Annotated[
    ObjectNDArray, Is[lambda arr: 1 <= arr.ndim <= 2],
]
