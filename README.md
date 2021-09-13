
<div align="center">
  <p>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide" >
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://github.com/moldyn/feature_selection/blob/main/LICENSE" alt="License" >
        <img src="https://img.shields.io/github/license/moldyn/feature_selection" /></a>
    <a href="https://moldyn.github.io/feature_selection" alt="Docs" >
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
- Simple CI

## Installation

```python
python3 -m pip install --upgrade feature_selection
```
or for the latest dev version
```python
python3 -m pip install git+https://github.com/moldyn/feature_selection.git
```

## Usage
### CI
The module brings a rich CI. Each module and submodule contains a rich help.
```bash
# creating correlation matrix
$ python -m cfs similarity -i input_file -o output_similarity -metric correlation -v

CFS SIMILARITY
~~~ Initialize similarity class
~~~ Load file input_file
~~~ Fit input
~~~ Store similarity matrix in output_similarity

# clustering with CPM and default resolution parameter
# the latter needs to be fine-tuned to each matrix
$ python -m cfs clustering -i output_similarity -o output_clustering -v

CFS CLUSTERING
~~~ Initialize clustering class
~~~ Load file output_similarity
~~~ Fit input
~~~ Store output
~~~ Plot matrix
```

### Inside Python script
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
