# Frequently Asked Questions

### How can I use the terminal user interface (TUI)?
This can be simply achieved by calling `python -m mosaic tui` or `mosaic tui`. It can be navigated via keyboard and mouse.


#### How to load the clusters file back to Python?
Simply use the function provided in `tools`:
```python
import mosaic

clusterfile = '...'
clusters = mosaic.tools.load_clusters(clusterfile)
```

#### Is it possible to use cross validation together with silhouette score?
The new release `v0.3.0` refactored the classes, so that the [`sklearn.model_selection.GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) can be used. Check out the [cv example](#cross-validation-of-parameters).


### Should I upgrade the package?
You can check out the CHANGELOG.md to see what changed.


### How can I interpretate the results?
Check out our publication for two detailed examples.


### Is it possible to install the CLI only?
Partially, yes. If you do not want to screw up your current Python environment there are multiples possibilities. Either create a virtual environment on your own via `conda` or `venv`, or you can simply use [pipx](https://pypa.github.io/pipx/)


### Is the silhouette method implemented?
Yes, simply use the `score` method implemented in the `Clustering` class.


### I want to use the UMAP embedding!
This is still possible if you switch to the version 0.3.2. We removed that feature in newer versions.


### What happened to the k-medoids clustering mode?
The k-medoids clustering mode (`mode='kmedoids'`) has been deprecated and removed in version 0.5.0. We recommend using one of the following clustering modes instead: `CPM`, `modularity`, or `linkage`. These modes provide more robust and efficient clustering capabilities for correlation-based feature selection.


### Feature X is missing
If you believe that a crucial functionality/method is missing, feel free to [open an issue](https://github.com/moldyn/MoSAIC/issues) and describe the missing functionality and why it should be added. Alternatively, you can implement it yourself and create a PR to add it to this package, see [contributing guide](../contributing).


### I found a bug. What to do next?
If you find a bug in this package, it is very kind of you to open an issue/bug report. This allows us to identify and fix the problem, thus improving the overall quality of the software for all users. By providing a clear and concise description of the problem, including steps to reproduce it, and relevant information such as device, operating system, and software version, you will help us resolve the problem quickly and effectively. Submitting a [bug report](https://github.com/moldyn/MoSAIC/issues) is a valuable contribution to the software and its community, and is greatly appreciated by the development team.
