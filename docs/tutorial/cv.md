
### Cross-Validation of Parameters
Selecting the optimal parameters, e.g., `resolution_parameter`, can be quite difficult. Here we show a short example how one can use cross-validation for optimizing the parameters. Nevertheless, one should keep in mind that the here used silhouette score is not optimal for our task.  Hence, the here obtained optimal parameters should be considered as a good first guess.

Here a figure visualizing the optimal resolution parameter and the code to produce it.

<img class="lightmode" style="width: 400px;" src="https://github.com/moldyn/MoSAIC/blob/main/docs/cv_silhouette_light.svg?raw=true#gh-light-mode-only" /><img class="darkmode" style="width: 400px;" src="https://github.com/moldyn/MoSAIC/blob/main/docs/cv_silhouette_dark.svg?raw=true#gh-dark-mode-only" />

```python
import mosaic
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt

pplt.use_style(colors='tab20c', figsize=2.4)

# traj = np.loadtxt(filename)

# specify parameters grid
resolution_params = np.linspace(0.1, 0.9, 20)
params = {'resolution_parameter': resolution_params}
search = mosaic.GridSearchCV(
    similarity=mosaic.Similarity(),
    clustering=mosaic.Clustering(
        mode='CPM',
        resolution_parameter=0.5,  # any dummy value is good here
    ),
    param_grid=params,
).fit(traj)

# plotting result
fig, ax = plt.subplots()
mean_score = search.cv_results_['mean_test_score']
std_score = search.cv_results_['std_test_score']

ax.fill_between(
    resolution_params,
    mean_score + std_score,
    mean_score - std_score,
    color='C2',
)
ax.plot(resolution_params, mean_score + std_score, c='C1')
ax.plot(resolution_params, mean_score - std_score, c='C1')
ax.plot(resolution_params, mean_score, c='C0')

ax.set_xlim([0, 1])
ax.set_xlabel(r'resolution parameter')
ax.set_ylabel(r'silhouette score')

pplt.savefig('cv_silhouette.pdf')
```
