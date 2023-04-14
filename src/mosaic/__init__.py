# -*- coding: utf-8 -*-
"""
MoSAIC is an advanced Python package specifically designed for the analysis of
discrete time series data from Molecular Dynamics (MD) simulations.
It offers a wide range of capabilities to identify collective coordinates that
describe the same biomolecular process.

With MoSAIC, researchers and engineers can easily analyze large and complex
datasets to gain insights into the underlying dynamics of biomolecular
processes. The package provides the capability to calculate various similarity
measures, such as linear correlation and mutual information, and to apply
different clustering algorithms to find groups of coordinates which move in a
concerted manner. By doing so, MoSAIC allows researchers and engineers to
identify groups of coordinates that collectively describe the same process in
MD simulations.

MoSAIC can be used as a stand-alone analysis tool or as a preprocessing step
for feature selection for subsequent Markov state modeling.
It is structured into the following submodules:

- [**similarity**][mosaic.similarity] This submodule introduces a versatile
class that enables the calculation of similarity measures based on different
correlation metrics. Users can choose from a set of popular metrics, such as
absolute value of Pearson correlation, of different normalizations of mutual
information. The result is always a similarity matrix,
which scales from 0 to 1. This submodule also supports efficient memory
management and flexible normalization options for mutual information-based
measures, making it a valuable addition to any data analysis pipeline.

- [**clustering**][mosaic.clustering] This submodule is the most central
component of MoSAIC that offers various techniques for analyzing similarity
matrices. It provides different modes for clustering a correlation matrix,
including the Leiden algorithm with different objective functions,
linkage clustering, and k-medoids, and supports both weighted and unweighted,
as well as full and sparse graphs.
The resulting clusters and labels can be accessed through the attributes of
the class.

- [**gridsearch**][mosaic.gridsearch] This submodule provides a class for
performing grid search cross validation. It allows users to explore different
combinations of parameter settings for a clustering model and provides
evaluation metrics for each combination. The best combination of parameters
and the corresponding model can be easily retrieved using the provided
attributes.

- [**utils**][mosaic.utils] This submodule provides utility functions that can
be used to store and load the data or to provide runtume user information.


"""
__all__ = ['Clustering', 'GridSearchCV', 'Similarity']

METRICS = {'correlation', 'NMI', 'JSD', 'GY'}
MODES = {'CPM', 'modularity', 'linkage', 'kmedoids'}
NORMS = {'joint', 'geometric', 'arithmetic', 'min', 'max'}

import mosaic.utils  # noqa: F401
from .clustering import Clustering
from .gridsearch import GridSearchCV
from .similarity import Similarity


__version__ = '0.4.1'
