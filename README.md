<div align="center">
  <img class="darkmode" style="width: 400px;" src="https://github.com/moldyn/MoSAIC/blob/main/docs/logo_large_dark.svg?raw=true#gh-dark-mode-only" />
  <img class="lightmode" style="width: 400px;" src="https://github.com/moldyn/MoSAIC/blob/main/docs/logo_large_light.svg?raw=true#gh-light-mode-only" />
  
  <p>
    <a href="https://github.com/wemake-services/wemake-python-styleguide" alt="wemake-python-styleguide">
        <img src="https://img.shields.io/badge/style-wemake-000000.svg" /></a>
    <a href="https://beartype.rtfd.io" alt="bear-ified">
        <img src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg" /></a>
    <a href="https://pypi.org/project/mosaic-clustering" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/mosaic-clustering" /></a>
    <a href="https://pepy.tech/project/mosaic-clustering" alt="Downloads">
        <img src="https://pepy.tech/badge/mosaic-clustering" /></a>
    <a href="https://github.com/moldyn/MoSAIC/actions/workflows/pytest.yml" alt="GitHub Workflow Status">
        <img src="https://img.shields.io/github/workflow/status/moldyn/MoSAIC/Pytest"></a>
    <a href="https://codecov.io/gh/moldyn/MoSAIC" alt="Code coverage">
        <img src="https://codecov.io/gh/moldyn/MoSAIC/branch/main/graph/badge.svg?token=KNWDAUXIGI" /></a>
    <a href="https://img.shields.io/pypi/pyversions/mosaic-clustering" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/mosaic-clustering" /></a>
    <a href="https://moldyn.github.io/MoSAIC" alt="Docs">
        <img src="https://img.shields.io/badge/pdoc3-Documentation-brightgreen" /></a>
    <a href="#" alt="doi">
        <img src="https://img.shields.io/badge/doi-submitted-blue" /></a>
    <a href="https://arxiv.org/abs/2204.02770" alt="arXiv">
        <img src="https://img.shields.io/badge/arXiv-2204.02770-red" /></a>
    <a href="https://github.com/moldyn/MoSAIC/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/moldyn/MoSAIC" /></a>
  </p>

  <p>
    <a href="https://moldyn.github.io/MoSAIC">Docs</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#faq">FAQ</a>
  </p>
</div>

# Molecular Systems Automated Identification of Cooperativity
MoSAIC is a new method for correlation analysis which automatically detects
collective motion in MD simulation data, identifies uncorrelated features
as noise and hence provides a detailed picture of the key coordinates driving a
conformational change in a biomolecular system. It is based on the Leiden community
detection algorithm which is used to bring a correlation matrix in a
block-diagonal form.

The method was published in:
> G. Diez, D. Nagel, and G. Stock,
> *Correlation-based feature selection to identify functional dynamcis
> in proteins*,
> [arXiv:2204.02770](https://arxiv.org/abs/2204.02770)

We kindly ask you to cite this article in case you use this software package for
published works.

## Features
- Intuitive usage via [module](#module---inside-a-python-script) and via [CI](#ci---usage-directly-from-the-command-line)
- Sklearn-style API for fast integration into your Python workflow
- No magic, only a  single parameter
- Extensive [documentation](https://moldyn.github.io/feature_selection) and
  detailed discussion in publication


## Installation
So far the package is only published to [PyPI](https://pypi.org). Soon, it will
be added [conda-forge](https://conda-forge.org/), as well. To install it within a python environment simple call:
```bash
python3 -m pip install --upgrade moldyn-mosaic
```
or for the latest dev version
```bash
# via ssh key
python3 -m pip install git+ssh://git@github.com/moldyn/MoSAIC.git

# or via password-based login
python3 -m pip install git+https://github.com/moldyn/MoSAIC.git
```

In case one wants to use the deprecated `UMAPSimilarity` or the module
`mosaic umap` one needs to specify the `extras_require='umap'`, so
```bash
python3 -m pip install --upgrade moldyn-mosaic[umap]
```

### Shell Completion
Using the `bash`, `zsh` or `fish` shell click provides an easy way to
provide shell completion, checkout the
[docs](https://click.palletsprojects.com/en/8.0.x/shell-completion).
In the case of bash you need to add following line to your `~/.bashrc`
```bash
eval "$(_MOSAIC_COMPLETE=bash_source mosaic)"
```

## Usage
In general one can call the module directly by its entry point `$ MoSAIC`
or by calling the module `$ python -m mosaic`. The latter method is
preferred to ensure using the desired python environment. For enabling
the shell completion, the entry point needs to be used.

### CI - Usage Directly from the Command Line
The module brings a rich CI using [click](https://click.palletsprojects.com).
Each module and submodule contains a detailed help, which can be accessed by
```bash
$ python -m mosaic
Usage: python -m mosaic [OPTIONS] COMMAND [ARGS]...

  MoSAIC motion v0.2.1

  Molecular systems automated identification of collective motion, is
  a correlation based feature selection framework for MD data.
  Copyright (c) 2021-2022, Georg Diez and Daniel Nagel

Options:
  --help  Show this message and exit.

Commands:
  clustering  Clustering similarity matrix of coordinates.
  similarity  Creating similarity matrix of coordinates.
  umap        Embedd similarity matrix with UMAP.
```
For more details on the submodule one needs to specify one of the three
commands.

A simple workflow example for clustering the input file `input_file` using
correlation and Leiden with CPM and the default resolution parameter:
```bash
# creating correlation matrix
$ python -m mosaic similarity -i input_file -o output_similarity -metric correlation -v

MoSAIC SIMILARITY
~~~ Initialize similarity class
~~~ Load file input_file
~~~ Fit input
~~~ Store similarity matrix in output_similarity

# clustering with CPM and default resolution parameter
# the latter needs to be fine-tuned to each matrix
$ python -m mosaic clustering -i output_similarity -o output_clustering --plot -v

MoSAIC CLUSTERING
~~~ Initialize clustering class
~~~ Load file output_similarity
~~~ Fit input
~~~ Store output
~~~ Plot matrix
```
This will generate the similarity matrix stored in `output_similarity`,
the plotted result in `output_clustering.matrix.pdf`, the raw data of
the matrix in `output_clustering.matrix` and a file containing in each
row the indices of a cluster.

### Module - Inside a Python Script
```python
import mosaic

# Load file
# X is np.ndarray of shape (n_samples, n_features)

sim = mosaic.Similarity(
    metric='correlation',  # or 'NMI', 'GY', 'JSD'
)
sim.fit(X)


# Cluster matrix
clust = mosaic.Clustering(
    mode='CPM',  # or 'modularity
)
clust.fit(sim.matrix_)

clusters = clust.clusters_
clusterd_X = clust.matrix_
...
```

### FAQ
#### How to load the clusters file back to Python?
Simply use the function provided in `tools`:
```python
import mosaic

clusterfile = '...'
clusters = mosaic.tools.load_clusters(clusterfile)
```

#### I get an error.
Please open an issue.

#### Should I upgrade the package?
You can check out the CHANGELOG.md to see what changed.

#### How can I interpretate the results?
Check out our publication for two detailed examples.

#### Is it possible to install the CLI only?
Partially, yes. If you do not want to screw up your current Python
environment there are multiples possibilities. Either create a
virtual environment on your own via `conda` or `venv`, or you can
simply use [pipx](https://pypa.github.io/pipx/)

#### Is the silhouette method implemented?
This feature will be added in `v0.3.0` in the next weeks.
