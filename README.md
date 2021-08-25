
<div align="center">
  <p>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide" >
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <!--
    <a href="https://pypi.org/project/feature_selection" alt="PyPI" >
        <img src="https://img.shields.io/pypi/v/cfs" /></a>
    <a href="https://pepy.tech/project/feature_selection" alt="Downloads" >
        <img src="https://pepy.tech/badge/feature_selection" /></a>
    <a href="https://img.shields.io/pypi/pyversions/feature_selection" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/feature_selection" /></a>
    -->
    <a href="https://github.com/moldyn/feature_selection/blob/main/LICENSE" alt="License" >
        <img src="https://img.shields.io/github/license/moldyn/feature_selection" /></a>
    <a href="https://braniii.gitlab.io/prettypyplot" alt="Doc" >
        <img src="https://img.shields.io/badge/pdoc3-Documentation-brightgreen" /></a>
    <a href="https://beartype.rtfd.io" alt="bear-ified" >
        <img src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg" /></a>
  </p>

  <p>
    <a href="https://moldyn.github.io/feature_selection">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a>
  </p>
</div>


# feature_selection
Correlation based feature selection of Molecular Dynamics simulations

## Features
...

## Usage
```python
import cfs

# Load file
# X is np.ndarray of shape (n_samples, n_features)

sim = cfs.Similarity(
    metric='correlation',  # or 'NMI', 'GY', 'JSD'
)
sim.fit(X)


# Cluster matrix
clust = cfs.Clustering(
    mode='CPM',  # or 'modularity
)
clust.fit(sim.matrix_)

clusters = clust.clusters_
clusterd_X = clust.matrix_
...
```

## Installation

```python
python3 -m pip install --upgrade feature_selection
```
or for the latest dev version
```python
python3 -m pip install git+https://github.com/moldyn/feature_selection.git
```
