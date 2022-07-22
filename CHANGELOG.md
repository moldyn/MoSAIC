# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and [Element](https://github.com/vector-im/element-android)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.3.0] - 2022-07-07
### API changes warning ⚠️:
- Module `tools` was renamed to `utils`
- `load_clusters` needs to be accessed via `mosaic.utils`
- Change the default precision from `half` to `single` in CI (`__main__.py`)
- Rename in `Similarity` class the parameter `online` to `low_memory` and in
  `mosaic similarity` the flag `--online` to `--low-memory`
- Rename in `Similarity` class the parameter `knn_estimator` to `use_knn_estimator`
- `Similarity.fit` and `Clustering.fit` are returning `self` which allows a
  chaining like `sim = Similarity().fit(X)`

### Added Features and Improvements 🙌:
- `Similarity` class derives now from `sklearn.base.BaseEstimator`
- `Clustering` class derives now from `sklearn.base.BaseEstimator` and `sklearn.base.BaseMixin`
- `fit_transform` and `transform`  method were added to `Similarity` class
- `fit_predict` method was added to `Clustering` class
- `score` method (for silhouette score) was added to `Clustering` class
- Added LGTM reports
- Allow to select the desired precision in the CI directly
- Improved clusters sorting algorithm to prefer larger clusters
- `mode='kmedoids'` was added to CI

### Bugfix 🐛:
- Fix bug of `load_clusters` in case of a single cluster
- Fix bug where `mode='kmedoids'` can not be executed with `n_clusters`
- Fix bug for `mode='kmedoids'`, preventing masking diagonal
- Prevent interpolation in plot
- Enforce higher precision on storing correlation matrix to remove artifacts
  of slight deviation from symmetry

### Other changes:
- Improved test functions of utils module
- Add more tests for clustering module
- Fixed some typos


## [0.2.2] - 2022-04-25
### Added Features and Improvements 🙌:
- Officially support python 3.10

### Other changes:
- Adding many helpful links to PyPI
- Remove dark-mode-only image from PyPI readme
- Add code coverage report via Codecov


## [0.2.1] - 2022-04-07
### API changes warning ⚠️:
- Removed `n_iterations` parameter from `Clustering`

### Added Features and Improvements 🙌:
- Added clustering mode `mode='kmedoids'` for k-medoids clustering
- Added tools module to reopen clusters

### Other changes:
- Improved some test functions


## [0.2.0] - 2022-04-01
### API changes warning ⚠️:
- Renamed cli script `mosaic`, former `MoSAIC`
- Renamed package to `mosaic-clustering`
- Renamed repository!!!

### Added Features and Improvements 🙌:
- Added MoSAIC logo
- Added changelog
- Unified extra requirements of module, use `[all], [umap], [docs], [testing]` with pip
- Add toy matrix of paper and script to generate similar matrices
- Improve html of docs
- Improve README

### Bugfix 🐛:
- Fix path of CLI script
- Fix coverage run
- Fix PEP 585 warnings by importing `typing` classes from `beartype.typing`
- Reset `Clustering.labels_` on calling `CLustering.fit(...)`
- Allow also numpy types, e.g., `np.float64`, or `np.int64` for resolution parameter

### Other changes:
- Increase required beartype version `0.8.1->0.10.4`
- Rename `DistanceMatrix` to `SimilarityMatrix`
- Fix example in `Clustering` to satisfy typing requirements
- Change import in `_typing.py` to non-private
- Improve some typings
- Require `typing_extensions` only for python < 3.9


## [0.1.1] - 2022-01-24
### Added Features and Improvements 🙌:
- Add `labels_` attribute to `Clustering`

### Bugfix 🐛:
- Use `metric='precomputed'` for UMAP
- Relax check of symmetric input by using `np.allclose()`
- Fix test expectation values

### Other changes:
- Catch warnings for umap


## [0.1.0] - 2021-12-09
- Initial release


[Unreleased]: https://github.com/moldyn/MoSAIC/compare/v0.3.0...main
[0.3.0]: https://github.com/moldyn/MoSAIC/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/moldyn/MoSAIC/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/moldyn/MoSAIC/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/moldyn/MoSAIC/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/moldyn/MoSAIC/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/moldyn/MoSAIC/tree/v0.1.0
