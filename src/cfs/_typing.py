# -*- coding: utf-8 -*-
"""Class for defining typing.

In this module all type hint types are defined. This includes as well the
available metrics, norms and clustering methods

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
from typing import List as _List
from typing import Union as _Union

import numpy as _np
from beartype.vale import Is as _Is

try:  # python <= 3.8
    from typing import Annotated as _Annotated
except ImportError:
    from typing_extensions import Annotated as _Annotated

# String (enum-type) datatypes
MetricString = _Annotated[
    str, _Is[lambda string: string in {'correlation', 'NMI', 'JSD', 'GY'}],
]
NormString = _Annotated[
    str,
    _Is[lambda string: string in {
        'joint', 'geometric', 'arithmetic', 'min', 'max',
    }],
]
ClusteringModeString = _Annotated[
    str, _Is[lambda string: string in {'CPM', 'modularity'}],
]

# scalar datatypes
PositiveInt = _Annotated[int, _Is[lambda val: val > 0]]
NumInRange0to1 = _Annotated[
    _Union[float, int], _Is[lambda val: 0 <= val <= 1],
]

# array datatypes
FloatNDArray = _Annotated[
    _np.ndarray, _Is[lambda arr: _np.issubdtype(arr.dtype, _np.floating)],
]
IntNDArray = _Annotated[
    _np.ndarray, _Is[lambda arr: _np.issubdtype(arr.dtype, _np.integer)],
]
ObjectNDArray = _Annotated[
    _np.ndarray, _Is[lambda arr: _np.issubdtype(arr.dtype, object)],
]
ArrayLikeFloat = _Union[_List[float], FloatNDArray]
Index1DArray = _Annotated[
    IntNDArray, _Is[
        lambda arr: arr.ndim == 1 and _np.all(arr >= 0)
    ],
]
Float1DArray = _Annotated[
    FloatNDArray, _Is[lambda arr: arr.ndim == 1],
]
Float2DArray = _Annotated[
    FloatNDArray, _Is[lambda arr: arr.ndim == 2],
]
FloatMatrix = _Annotated[
    Float2DArray,
    _Is[lambda arr: arr.shape[0] == arr.shape[1]],
]
FloatMax2DArray = _Annotated[
    FloatNDArray, _Is[lambda arr: 1 <= arr.ndim <= 2],
]
Object1DArray = _Annotated[
    ObjectNDArray, _Is[lambda arr: arr.ndim == 1],
]
ObjectMax2DArray = _Annotated[
    ObjectNDArray, _Is[lambda arr: 1 <= arr.ndim <= 2],
]
