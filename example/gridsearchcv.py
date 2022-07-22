import mosaic
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt

pplt.use_style(colors='tab20c', figsize=2.4)


def main():
    ## Load trajectory from file
    # traj = np.loadtxt(filename)
    ## Here we use some random sample data
    traj = create_traj()

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
        color='C0',
    )
    ax.plot(n_clusters, mean_score + std_score, c='C1')
    ax.plot(n_clusters, mean_score - std_score, c='C1')
    ax.plot(n_clusters, mean_score, c='C2')

    ax.set_xlim([0, traj.shape[1]])
    ax.set_xlabel(r'$k$ no. of clusters')
    ax.set_ylabel(r'silhouette score')


    pplt.savefig('cv_silhouette.pdf')


def create_traj():
    """Creating sample trajectory."""
    np.random.seed(42)

    x = np.linspace(0, 2 * np.pi, 1000)
    rand_offsets = np.random.uniform(
        low=-np.pi / 6, high=np.pi / 6, size=10,
    )

    traj = np.array([
        *[np.sin(x + xi) for xi in rand_offsets],
        *[np.cos(x + xi) for xi in rand_offsets],
        *[np.zeros_like(x) for xi in rand_offsets],
    ]).T
    return traj + np.random.normal(size=traj.shape, scale=.2)


if __name__ == '__main__':
    main()
