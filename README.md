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
    <a href="https://anaconda.org/conda-forge/mosaic-clustering" alt="conda version">
	<img src="https://img.shields.io/conda/vn/conda-forge/mosaic-clustering" /></a>
    <a href="https://pepy.tech/project/mosaic-clustering" alt="Downloads">
        <img src="https://pepy.tech/badge/mosaic-clustering" /></a>
    <a href="https://github.com/moldyn/MoSAIC/actions/workflows/pytest.yml" alt="GitHub Workflow Status">
        <img src="https://img.shields.io/github/workflow/status/moldyn/MoSAIC/Pytest"></a>
    <a href="https://codecov.io/gh/moldyn/MoSAIC" alt="Code coverage">
        <img src="https://codecov.io/gh/moldyn/MoSAIC/branch/main/graph/badge.svg?token=KNWDAUXIGI" /></a>
    <a href="https://lgtm.com/projects/g/moldyn/MoSAIC" alt="Code coverage">
	<img src="https://img.shields.io/lgtm/grade/python/github/moldyn/MoSAIC" alt="LGTM Grade" /></a>
    <a href="https://img.shields.io/pypi/pyversions/mosaic-clustering" alt="PyPI - Python Version">
        <img src="https://img.shields.io/pypi/pyversions/mosaic-clustering" /></a>
    <a href="https://moldyn.github.io/MoSAIC" alt="Docs">
        <img src="https://img.shields.io/badge/pdoc3-Documentation-brightgreen" /></a>
    <a href="https://doi.org/10.1021/acs.jctc.2c00337" alt="doi">
        <img src="https://img.shields.io/badge/doi-10.1021%2Facs.jctc.2c00337-blue" /></a>
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
MoSAIC is an unsupervised method for correlation analysis which automatically detects
the collective motion in MD simulation data, while simultaneously identifying
uncorrelated coordinates as noise.
Hence, it can be used as a feature selection scheme for Markov state modeling or simply
to obtain a detailed picture of the key coordinates driving a biomolecular process.
It is based on the Leiden community detection algorithm which is used to bring a
correlation matrix in a block-diagonal form.

The method was published in:
> **Correlation-Based Feature Selection to Identify Functional Dynamcis
> in Proteins**  
> G. Diez, D. Nagel, and G. Stock,
> *J. Chem. Theory Comput.* **2022** 18 (8), 5079-5088,
> [10.1021/acs.jctc.2c00337](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00337)

If you use this software package, pleaes cite the above mentioned paper.

## Features
- Intuitive usage via [module](#module---inside-a-python-script) and via
  [CI](#ci---usage-directly-from-the-command-line)
- Sklearn-style API for fast integration into your Python workflow
- No magic, only a  single parameter which can be optimized via
  cross-validation
- Extensive [documentation](https://moldyn.github.io/MoSAIC) and
  detailed discussion in publication


## Installation
The package is called `mosaic-clustering` and is available via
[PyPI](https://pypi.org/project/mosaic-clustering) or
[conda](https://anaconda.org/conda-forge/mosaic-clustering). To install it,
simply call:
```bash
python3 -m pip install --upgrade mosaic-clustering
```
or
```
conda install -c conda-forge mosaic-clustering
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

  MoSAIC motion v0.3.1

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

### Cross-Validation of Parameters
Selecting the optimal parameters, e.g., `resolution_parameter`, or
`n_clusters`, can be quite difficult. Here we show a short example how one can
use cross-validation for optimizing the parameters. Nevertheless, one should
keep in mind that the here used silhouette score is not optimal for our task.
Hence, the here obtained optimal parameters should be considered as a good
first guess.

Here a figure visualizing the optimal cluster value `n_clusters=12` and the
code to produce it.

<img class="lightmode" style="width: 400px;" src="https://github.com/moldyn/MoSAIC/blob/main/docs/cv_silhouette_light.svg?raw=true#gh-light-mode-only" /><img class="darkmode" style="width: 400px;" src="https://github.com/moldyn/MoSAIC/blob/main/docs/cv_silhouette_dark.svg?raw=true#gh-dark-mode-only" />

```python
import mosaic
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt

pplt.use_style(colors='tab20c', figsize=2.4)

# traj = np.loadtxt(filename)

# specify parameters grid
n_clusters = np.arange(2, traj.shape[1])
params = {'n_clusters': n_clusters}
search = mosaic.GridSearchCV(
    similarity=mosaic.Similarity(),
    clustering=mosaic.Clustering(
        mode='kmedoids',
        n_clusters=2,  # any dummy value is good here
    ),
    param_grid=params,
).fit(traj)

# plotting result
fig, ax = plt.subplots()
mean_score = search.cv_results_['mean_test_score']
std_score = search.cv_results_['std_test_score']

ax.fill_between(
    n_clusters,
    mean_score + std_score,
    mean_score - std_score,
    color='C2',
)
ax.plot(n_clusters, mean_score + std_score, c='C1')
ax.plot(n_clusters, mean_score - std_score, c='C1')
ax.plot(n_clusters, mean_score, c='C0')

ax.set_xlim([0, traj.shape[1]])
ax.set_xlabel(r'$k$ no. of clusters')
ax.set_ylabel(r'silhouette score')

pplt.savefig('cv_silhouette.pdf')
```

### FAQ
#### How to load the clusters file back to Python?
Simply use the function provided in `tools`:
```python
import mosaic

clusterfile = '...'
clusters = mosaic.tools.load_clusters(clusterfile)
```

#### Is it possible to use cross validation together with silhouette score?
The new release `v0.3.0` refactored the classes, so that the
[`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
can be used. Check out the [cv example](#cross-validation-of-parameters).

#### I get an error.
Please [open an issue](https://github.com/moldyn/MoSAIC/issues/new/choose).

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
Yes, simply use the `score` method implemented in the `Clustering` class.
