# -*- coding: utf-8 -*-
"""Methods for calculating linear and non-linear correlation measures.

MIT License
Copyright (c) 2022, Daniel Nagel
All rights reserved.

"""
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Generator, Optional, Tuple, Union
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_regression

from mosaic._typing import (  # noqa: WPS436
    ArrayLikeFloat,
    DTypeLike,
    Float1DArray,
    Float2DArray,
    FloatMatrix,
    NormString,
    PositiveInt,
)


@beartype
def _freedman_diaconis_rule(x: Float1DArray) -> int:
    """Freedman Diaconis rule to estimate number of bins.

    We replaced the prefactor 2 by 2.59 to asymptotically match Scott's normal
    reference rule for a normal distribution. See
    https://www.wikiwand.com/en/Freedman–Diaconis_rule

    Freedman, David and Diaconis, Persi (1981):
    "On the histogram as a density estimator: L2 theory"
    Probability Theory and Related Fields. 57 (4), 453–476

    """
    scott_factor = 2.59
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    binwidth = scott_factor * iqr / np.cbrt(len(x))
    ptp = np.min([
        np.subtract(*np.percentile(x, [85, 0])),
        np.subtract(*np.percentile(x, [100, 15])),
    ])
    return int(
        np.ceil(ptp / binwidth),
    )


@beartype
def _entropy(p: ArrayLikeFloat) -> float:
    """Calculate entropy of density p."""
    return -1 * np.sum(p * np.ma.log(p))


@beartype
def _kullback(p: ArrayLikeFloat, q: ArrayLikeFloat) -> float:
    """Calculate Kullback-Leibler divergence of density p, q."""
    if len(p) != len(q):
        raise ValueError(
            f'Arrays p, q need to be of same length, but {len(p):.0f} vs '
            f'{len(q):.0f}.',
        )
    return np.sum(
        p * np.ma.log(np.ma.divide(p, q)),
    )


@beartype
def _estimate_densities(
    x: Float1DArray, y: Float1DArray, bins: Optional[PositiveInt] = None,
) -> Tuple[Float2DArray, Float2DArray, Float1DArray, Float1DArray]:
    """Calculate two dimensional probability densities."""
    if bins is None:
        bins = [
            _freedman_diaconis_rule(x),
            _freedman_diaconis_rule(y),
        ]
    hist, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    # transpose since numpy considers axis 0 as y and axis 1 as x
    pxy = hist.T / np.sum(hist)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    pxpy = px[:, np.newaxis] * py[np.newaxis, :]

    return pxy, pxpy, px, py


@beartype
def _correlation(X: Float2DArray) -> FloatMatrix:
    """Return the correlation of input.

    Each feature (column) of X need to be mean-free with standard deviation 1.

    """
    corr = (
        X.T / len(X) @ X
    ).astype(np.float64)

    # symmetrize and enforce diag equals 1
    if not np.allclose(corr, corr.T):
        raise ValueError(
            'Correlation matrix is not symmetric. This should not occur and '
            'is probably caused by an overflow error.'
        )
    if not np.allclose(np.diag(corr), 1):
        raise ValueError(
            'Self-correlation is not 1. This should not occur and is probably'
            'caused by an overflow error.'
        )
    corr = 0.5 * (corr + corr.T)
    corr[np.diag_indices_from(corr)] = 1
    return corr


@beartype
def _knn_mutual_information(
    X: Float2DArray,
) -> FloatMatrix:
    """Return the knn-estimated mutual information of input."""
    n_features: PositiveInt
    _, n_features = X.shape
    mi_knn: FloatMatrix = np.empty((n_features, n_features))
    for idx_feature, feature in enumerate(X.T):
        mi_knn[idx_feature] = mutual_info_regression(
            X, feature,
        )
    return mi_knn


@beartype
def _nonlinear_GY_knn(X: Float2DArray) -> FloatMatrix:
    """Gel'fand-Yaglom correlation matrix based on k-nn estimator.

    Return the nonlinear correlation matrix based on Gel'fand-Yaglom
    based on a reliable knn-estimate of the mutual information.

    """
    nl_knn_corr = _knn_mutual_information(X)
    return np.sqrt(
        1 - np.exp(-2 * nl_knn_corr),
    )


@beartype
def _gy(
    pij: Float2DArray,
    pipj: Float2DArray,
    pi: Float1DArray,
    pj: Float1DArray,
) -> float:
    """Return the dissimilarity based on Gel'fand-Yaglom."""
    mutual_info: float = _kullback(pij, pipj)
    return np.sqrt(
        1 - np.exp(-2 * mutual_info),
    )


@beartype
def _jsd(
    pij: Float2DArray,
    pipj: Float2DArray,
    pi: Float1DArray,
    pj: Float1DArray,
) -> float:
    """Return the Jensen-Shannon based dissimilarity."""
    return jensenshannon(
        pij.flatten(),
        pipj.flatten(),
        base=2,
    )


@beartype
def _normalization(
    pi: np.ndarray, pj: np.ndarray, pij: np.ndarray, method: NormString,
) -> float:
    """Calculate the normalization factor for the MI matrix."""
    if method == 'joint':
        return _entropy(pij)

    func: Callable = {
        'geometric': lambda arr: np.sqrt(np.prod(arr)),
        'arithmetic': np.mean,
        'min': np.min,
        'max': np.max,
    }[method]
    return func([_entropy(pi), _entropy(pj)])


@beartype
def _nmi_gen(method: str) -> Callable:
    @beartype
    def _nmi(  # noqa: WPS430
        pij: Float2DArray,
        pipj: Float2DArray,
        pi: Float1DArray,
        pj: Float1DArray,
    ) -> float:
        """Return the Jensen-Shannon based dissimilarity."""
        mutual_info: float = _kullback(pij, pipj)
        normalization: float = _normalization(pi, pj, pij, method=method)
        return mutual_info / normalization

    return _nmi


@beartype
def _data_gen(
    *,
    filename: str,
    dtype: DTypeLike,
    comments: Union[str, Tuple[str, ...]] = ('#', '@'),
) -> Generator[Float1DArray, None, None]:
    """Return all non comment lines as generator."""
    with open(filename) as file_obj:
        for line in file_obj:
            if line.startswith(comments):
                continue
            yield np.array(line.split()).astype(dtype)


@beartype
def _get_n_features(
    *,
    filename: str,
    dtype: DTypeLike,
) -> int:
    """Return the number of columns/features."""
    return len(
        next(
            _data_gen(filename=filename, dtype=dtype),
        ),
    )


@beartype
def _welford_correlation(
    *,
    filename: str,
    dtype: DTypeLike,
) -> Tuple[FloatMatrix, int, int]:
    """Calculate the correlation via online Welford algorithm.

    Welford algorithm, generalized to correlation. Taken from:
    Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
    Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.

    """
    n_samples: int = 0
    n_features: int = _get_n_features(filename=filename, dtype=dtype)
    mean: np.ndarray = np.zeros(n_features, dtype=dtype)
    corr: np.ndarray = np.zeros(  # noqa: WPS317
        (n_features, n_features), dtype=dtype,
    )

    for x in _data_gen(filename=filename, dtype=dtype):
        n_samples += 1
        dx: np.ndarray = x - mean
        mean = mean + dx / n_samples
        corr = corr + dx.reshape(-1, 1) * (
            x - mean
        ).reshape(1, -1)

    if n_samples < 2:
        return np.full_like(corr, np.nan), n_samples, n_features

    std = np.sqrt(np.diag(corr) / (n_samples - 1))
    corr = corr / (n_samples - 1) / (
        std.reshape(-1, 1) * std.reshape(1, -1)
    )

    return corr, n_samples, n_features
