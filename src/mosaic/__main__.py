"""CLI of MoSAIC.

MIT License
Copyright (c) 2021-2022, Daniel Nagel
All rights reserved.

"""
import click
import numpy as np
import pandas as pd
import prettypyplot as pplt
from matplotlib import pyplot as plt

import mosaic
from mosaic.utils import save_clusters, savetxt

# setup matplotlibs rcParam
pplt.use_style(figsize=2, figratio=1, cmap='turbo')

NORMALIZES = ['joint', 'geometric', 'arithmetic', 'min', 'max']
METRICS = ['correlation', 'NMI', 'JSD', 'GY']
MODES = ['CPM', 'modularity', 'linkage', 'kmedoids']
PRECISION = ['half', 'single', 'double']
PRECISION_TO_DTYPE = {
    'half': np.float16,
    'single': np.float32,
    'double': np.float64,
}

HELP_STR = f"""MoSAIC motion v{mosaic.__version__}

\b
Molecular systems automated identification of collective motion, is
a correlation based feature selection framework for MD data.
Copyright (c) 2021-2022, Georg Diez and Daniel Nagel
"""


@click.group(help=HELP_STR)
def main():
    """Empty group to show on help available submodules."""
    pass


@main.command(
    help='Creating similarity matrix of coordinates.',
    no_args_is_help=True,
)
@click.option(
    '--metric',
    default='correlation',
    show_default=True,
    type=click.Choice(METRICS, case_sensitive=True),
    help='Metric used to estimate similarity measure matrix.',
)
@click.option(
    '--normalize-method',
    type=click.Choice(NORMALIZES, case_sensitive=True),
    help=(
        'Only required for metric="NMI". Determines the normalization factor '
        'for the mutual information. See docs for help.'
    ),
)
@click.option(
    '--low-memory',
    is_flag=True,
    help=(
        'If set the correlation is calculated on-the-fly using the online '
        'Welford algorithm. This is much slower but needs less RAM and is '
        'preferable for larger files. This supports only metric=correlation.'
    ),
)
@click.option(
    '-i',
    '--input',
    'input_file',
    required=True,
    type=click.Path(exists=True),
    help=(
        'Path to input file. Needs to be of shape (n_samples, n_features).'
        ' All command lines need to start with "#". By default np.float16'
        ' is used for the datatype.'
    ),
)
@click.option(
    '-o',
    '--output',
    'output_file',
    required=True,
    type=click.Path(),
    help=(
        'Path to output file. Will be a matrix of shape (n_features, '
        'n_features).'
    ),
)
@click.option(
    '--knn_estimator',
    is_flag=True,
    default=False,
    help=(
        'Uses a parameter free estimate for the Gelfand-Yaglom mutual'
        'information based distance measure which yields more accurate'
        'results, but is computationally more expensive.'
    )
)
@click.option(
    '--precision',
    default='single',
    show_default=True,
    type=click.Choice(PRECISION, case_sensitive=True),
    help=(
        'Precision used for calculation. Lower precision reduces memory '
        'impact but may lead to overflow errors.'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Activate verbose mode.',
)
def similarity(
    metric,
    low_memory,
    normalize_method,
    input_file,
    output_file,
    knn_estimator,
    precision,
    verbose,
):
    if verbose:
        click.echo('\nMoSAIC SIMILARITY\n~~~ Initialize similarity class')
    sim = mosaic.Similarity(
        metric=metric,
        low_memory=low_memory,
        normalize_method=normalize_method,
        use_knn_estimator=knn_estimator,
    )
    if low_memory:
        if verbose:
            click.echo(f'~~~ Fit on-the-fly {input_file}')
        sim.fit(input_file)
    else:
        if verbose:
            click.echo(f'~~~ Load file {input_file}')
        X = pd.read_csv(
            input_file,
            sep=r'\s+',
            header=None,
            comment='#',
            dtype=PRECISION_TO_DTYPE[precision],
        ).values
        if verbose:
            click.echo('~~~ Fit input.')
        sim.fit(X)

    if verbose:
        click.echo(f'~~~ Store similarity matrix in {output_file}')
    savetxt(
        output_file,
        sim.matrix_,
        fmt='%.5f',
        submodule='similarity',
        header='Similarity matrix',
    )


@main.command(
    help='Clustering similarity matrix of coordinates.',
    no_args_is_help=True,
)
@click.option(
    '--mode',
    default='CPM',
    show_default=True,
    type=click.Choice(MODES, case_sensitive=True),
    help='Mode used for Leiden clustering.',
)
@click.option(
    '--n-neighbors',
    type=click.IntRange(min=2),
    help=(
        'If unequal to None, a knn-graph will be used. If None, for mode "CPM"'
        ' the whole matrix is used, while for "modularity" the '
        'sqrt(n_features)'
    ),
)
@click.option(
    '--resolution-parameter',
    type=click.FloatRange(min=0, max=1),
    help='Resolution parameter used for CPM.',
)
@click.option(
    '--n-clusters',
    type=click.IntRange(min=2),
    help='Required for mode="kmedoids". The number of clusters to form.',
)
@click.option(
    '-i',
    '--input',
    'input_file',
    required=True,
    type=click.Path(exists=True),
    help='Path to input file. Needs to be of shape (n_features, n_features).',
)
@click.option(
    '-o',
    '--output-basename',
    'output_file',
    type=click.Path(),
    help='Basename of output files.',
)
@click.option(
    '--weighted/--unweighted',
    is_flag=True,
    default=True,
    show_default=True,
    help='Using an adjacency graph (not supported for CPM).',
)
@click.option(
    '--plot',
    is_flag=True,
    help='Plotting matrix.',
)
@click.option(
    '-n',
    '--name',
    'name_file',
    type=click.Path(exists=True),
    help=(
        'Path to file containing names of each colum. '
        'Needs to be of shape (n_features, ).'
    ),
)
@click.option(
    '--precision',
    default='single',
    show_default=True,
    type=click.Choice(PRECISION, case_sensitive=True),
    help=(
        'Precision used for calculation. Lower precision reduces memory '
        'impact but may lead to overflow errors.'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Activate verbose mode.',
)
def clustering(
    mode,
    input_file,
    n_neighbors,
    resolution_parameter,
    n_clusters,
    weighted,
    output_file,
    name_file,
    plot,
    precision,
    verbose,
):
    if verbose:
        click.echo('\nMoSAIC CLUSTERING\n~~~ Initialize clustering class')

    if not output_file:
        output_file = input_file

    clust = mosaic.Clustering(
        mode=mode,
        weighted=weighted,
        n_neighbors=n_neighbors,
        n_clusters=n_clusters,
        resolution_parameter=resolution_parameter,
    )

    if verbose:
        click.echo(f'~~~ Load file {input_file}')
    X = pd.read_csv(
        input_file,
        sep=r'\s+',
        header=None,
        comment='#',
        dtype=PRECISION_TO_DTYPE[precision],
    ).values
    X = np.abs(X)
    if verbose:
        click.echo('~~~ Fit input')
    clust.fit(X)

    if verbose:
        click.echo('~~~ Store output')
    savetxt(
        f'{output_file}.matrix',
        clust.matrix_,
        fmt='%.5f',
        submodule='clustering',
        header=(
            f'Permuted similarity matrix. In file {output_file}.clusters '
            'the corresponding clusters are listed.'
        ),
    )

    save_clusters(
        f'{output_file}.clusters',
        clust.clusters_,
    )

    if name_file:
        names = np.loadtxt(name_file, dtype=str)
        clusters_string = np.array(
            [
                ' '.join([names[state] for state in cluster])
                for cluster in clust.clusters_
            ],
            dtype=str,
        )
        savetxt(
            f'{output_file}.cluster_names',
            clusters_string,
            fmt='%s',
            submodule='clustering',
            header=(
                'In ith row are the names corresponding to cluster i.'
            ),
        )

    if plot:
        if verbose:
            click.echo('~~~ Plot matrix')

        _, ax = plt.subplots()
        mat = clust.matrix_.astype(np.float64)
        mat[np.diag_indices_from(mat)] = np.nan
        im = ax.imshow(
            mat, aspect='equal', origin='upper', interpolation='none',
        )

        ticks = np.array([0, *clust.ticks_[: -1]]) - 0.5
        major_mask = np.array([
            len(cluster) > 2 for cluster in clust.clusters_
        ])
        ticklabels = np.arange(len(ticks)) + 1
        for set_ticks, set_ticklabels in (
            (ax.set_xticks, ax.set_xticklabels),
            (ax.set_yticks, ax.set_yticklabels),
        ):
            set_ticks(ticks[major_mask])
            set_ticklabels(ticklabels[major_mask])
            set_ticks(ticks[~major_mask], minor=True)
            set_ticklabels([], minor=True)

        ax.grid(b=True, ls='-', lw=0.5)
        ax.grid(b=True, ls='-', which='minor', lw=0.1)

        ax.set_xlabel('clusters')
        ax.set_ylabel('clusters')

        pplt.colorbar(im, width='3%')
        pplt.savefig(f'{output_file}.matrix.pdf')


@main.command(
    help='Embedd similarity matrix with UMAP.',
    no_args_is_help=True,
)
@click.option(
    '--n-components',
    default=mosaic.UMAPSimilarity._default_n_components,  # noqa: WPS437
    show_default=True,
    type=click.IntRange(min=2),
    help='Dimensionality of UMAP embedding.',
)
@click.option(
    '--n-neighbors',
    type=click.IntRange(min=2),
    help=(
        'Number of nearest neighbors used for estimating manifold. '
        'If None, the sqrt of no. of features is used.'
    ),
)
@click.option(
    '--densmap/--umap',
    default=True,
    is_flag=True,
    show_default=True,
)
@click.option(
    '-i',
    '--input',
    'input_file',
    required=True,
    type=click.Path(exists=True),
    help=(
        'Path to input file. Needs to be of shape (n_features, n_features).'
        ' All command lines need to start with "#".'
    ),
)
@click.option(
    '-o',
    '--output',
    'output_file',
    required=True,
    type=click.Path(),
    help=(
        'Path to output file. Will be a matrix of shape (n_features, '
        'n_features).'
    ),
)
@click.option(
    '--precision',
    default='single',
    show_default=True,
    type=click.Choice(PRECISION, case_sensitive=True),
    help=(
        'Precision used for calculation. Lower precision reduces memory '
        'impact but may lead to overflow errors.'
    ),
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Activate verbose mode.',
)
def umap(
    n_components,
    n_neighbors,
    densmap,
    input_file,
    output_file,
    precision,
    verbose,
):
    if verbose:
        click.echo('\nMoSAIC UMAP\n~~~ Initialize umap similarity class')
    umapsim = mosaic.UMAPSimilarity(
        densmap=densmap,
        n_neighbors=n_neighbors,
        n_components=n_components,
    )
    if verbose:
        click.echo(f'~~~ Load file {input_file}')
    X = pd.read_csv(
        input_file,
        sep=r'\s+',
        header=None,
        comment='#',
        dtype=PRECISION_TO_DTYPE[precision],
    ).values
    if verbose:
        click.echo('~~~ Fit input.')
    umapsim.fit(X)
    if verbose:
        click.echo(f'~~~ Store similarity matrix in {output_file}')
    savetxt(
        output_file,
        umapsim.matrix_,
        fmt='%.5f',
        submodule='umap',
        header='Similarity matrix',
    )


if __name__ == '__main__':
    main()
