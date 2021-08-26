# -*- coding: utf-8 -*-
"""Script for generating toy model.

Taken from: Ravindra et al., "Automatic mutual information noise ommission
(AMINO): generating order paramters for molecular systems", Mol. Syst. Des.
Eng., 2020, 5, 339-348.

MIT License
Copyright (c) 2021, Daniel Nagel
All rights reserved.

"""
import numpy as np

NSAMPLES = 100000


def gen_two(samples):
    """Generate two state model coordinate."""
    return (
        1 - np.random.uniform(size=samples)**4
    ) * np.random.choice([-1, 1], size=samples)


def gen_three(samples):
    """Generate three state model coordinate."""
    traj = np.random.uniform(size=samples)**4 / 2

    rand_selection = np.random.choice([True, False], size=samples)
    traj[rand_selection] = 1 - traj[rand_selection]

    rand_selection = np.random.choice([True, False], size=samples)
    traj[rand_selection] *= -1
    return traj


def noise(samples, width=0.1):
    """Generate noise coordinate."""
    return np.random.uniform(-0.5 * width, 0.5 * width, size=samples)


def make_amino_ops():
    """Generate AMINO toy model coordinates."""
    ops = np.empty((NSAMPLES, 120))
    idx = 0
    for ndubs, gen in (
        (14, gen_two),
        (14, gen_two),
        (14, gen_two),
        (10, gen_two),
        (10, gen_two),
        (10, gen_two),
        (14, gen_three),
        (14, gen_three),
        (10, gen_three),
        (10, gen_three),
    ):
        ops[:, idx] = gen(NSAMPLES)
        for i in range(idx + 1, idx + ndubs):
            ops[:, i] = ops[:, idx] + noise(NSAMPLES, 0.1)
        idx += ndubs

    np.savetxt(
        f'amino_120d_{NSAMPLES}',
        ops,
        header=(
            'Toy example reproduced from Ravindra et al., "Automatic mutual '
            'information noise ommission (AMINO): generating order paramters '
            'for molecular systems", Mol. Syst. Des. Eng., 2020, 5, 339-348.'
        ),
        fmt='%.4f',
    )


if __name__ == '__main__':
    # fix random seed to make dataset reproducible
    magic_number = 42
    np.random.seed(magic_number)

    make_amino_ops()
