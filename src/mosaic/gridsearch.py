# -*- coding: utf-8 -*-
"""Class for GridSearchCV with silhouette score.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
__all__ = ['GridSearchCV']  # noqa: WPS410

import numpy as np
from beartype import beartype
from beartype.typing import Dict, Optional
from sklearn.model_selection import GridSearchCV as SKGridSearchCV
from sklearn.pipeline import Pipeline

from mosaic.clustering import Clustering
from mosaic.similarity import Similarity
from mosaic._typing import FloatMax2DArray  # noqa: WPS436


class GridSearchCV(SKGridSearchCV):
    r"""Class for grid searhc cross validation.

    Parameters
    ----------
    similarity : mosaic.Similarity
        Similarity instance setup with constant parameters, see
        `mosaic.Similarity` for available parameters. `low_memory` is not
        supported.

    clustering : mosaic.Clustering
        Clustering instance setup with constant parameters, see
        `mosaic.Clustering` for available parameters.

    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored.

    gridsearch_kwargs : dict
        Dictionary with parameters to be used for
        [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
        class. The parameter `estimator` is not supported and `param_grid`
        needs to be passed directly to the class.


    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns.

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the `cv_results_` arrays) which corresponds to the best
        candidate parameter setting.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    Check out
    [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    for an overview of all available attributes and more detailed description.

    Examples
    --------
    >>> import mosaic
    >>> # create two correlated data sets
    >>> traj = np.array([
    ...     func(np.linspace(0, 20, 1000))
    ...     for  func in (
    ...         np.sin,
    ...         lambda x: np.sin(x + 0.1),
    ...         np.cos,
    ...         lambda x: np.cos(x + 0.1),
    ...     )
    ... ]).T
    >>> search = mosaic.GridSearchCV(
    ...     similarity=mosaic.Similarity(),
    ...     clustering=mosaic.Clustering(),
    ...     param_grid={'resolution_parameter': [0.05, 0.2]},
    ... )
    >>> search.fit(traj)
    GridSearchCV(clustering=Clustering(),
                 param_grid={'clust__resolution_parameter': [0.05, 0.2]},
                 similarity=Similarity())
    >>> search.best_params_
    {'clust__resolution_parameter': 0.2}
    >>> search.best_estimator_
    Pipeline(steps=[('sim', Similarity()),
                    ('clust', Clustering(resolution_parameter=0.2))])

    """

    _sim_prefix: str = 'sim'
    _clust_prefix: str = 'clust'

    @beartype
    def __init__(
        self,
        *,
        similarity: Similarity,
        clustering: Clustering,
        param_grid: Dict,
        gridsearch_kwargs: Dict = {},
    ) -> None:
        """Initialize GridSearchCV class."""
        self.similarity: Similarity = similarity
        self.clustering: Clustering = clustering
        self.gridsearch_kwargs: Dict = gridsearch_kwargs

        if 'estimator' in self.gridsearch_kwargs:
            raise NotImplementedError(
                'Custom estimators are not supported. Please use the '
                'sklearn class GirdSearchCV directly.',
            )

        if 'param_grid' in self.gridsearch_kwargs:
            raise NotImplementedError(
                "Please pass 'param_grid' directly to the the class.",
            )

        if similarity.get_params()['low_memory']:
            raise NotImplementedError(
                "'low_memory' is currently not implemented.",
            )

        if not param_grid:
            raise ValueError(
                'At least a single parameter needs to be provided',
            )

        self.pipeline = Pipeline([
            (self._sim_prefix, self.similarity),
            (self._clust_prefix, self.clustering),
        ])

        self.param_grid: Dict = {}
        for param, values in param_grid.items():
            if param in similarity.get_params():
                self.param_grid[
                    f'{self._sim_prefix}__{param}'
                ] = values
            elif param in clustering.get_params():
                self.param_grid[
                    f'{self._clust_prefix}__{param}'
                ] = values
            else:
                raise ValueError(
                    f"param_grid key '{param}' is not available."
                )

        super().__init__(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            **self.gridsearch_kwargs,
        )

    @beartype
    def fit(
        self,
        X: FloatMax2DArray,
        y: Optional[np.ndarray] = None,
    ):
        """Clusters the correlation matrix by Leiden clustering on a graph.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : Ignored
            Not used, present for scikit API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        return super().fit(X)
