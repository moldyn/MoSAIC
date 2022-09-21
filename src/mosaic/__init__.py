# -*- coding: utf-8 -*-
""".. include:: ../../README.md"""
__all__ = ['Clustering', 'GridSearchCV', 'Similarity', 'UMAPSimilarity']

METRICS = {'correlation', 'NMI', 'JSD', 'GY'}
MODES = {'CPM', 'modularity', 'linkage', 'kmedoids'}
NORMS = {'joint', 'geometric', 'arithmetic', 'min', 'max'}

import mosaic.utils  # noqa: F401
from .clustering import Clustering
from .gridsearch import GridSearchCV
from .similarity import Similarity
from .umap_similarity import UMAPSimilarity


__version__ = '0.3.2'
