from distributional_learner import run
from os import path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def sample_inputs(mu_file, cov_file, counts_file, dimensions, num_samples=5000):
    mus = pd.read_csv(mu_file)
    covs = pd.read_csv(cov_file)
    counts = pd.read_csv(counts_file)

    mu_names = ['vowel'] + dimensions
    mus = mus.filter(mu_names)

    cov_names = ['vowel'] + [
        x for x in covs.columns 
        if x.split('-')[0] in dimensions and x.split('-')[1] in dimensions
    ]
    covs = covs.filter(cov_names)

    dists = []

    for v in mus.vowel:
        dists.append(
            torch.distributions.MultivariateNormal(
                torch.Tensor(mus[mus.vowel == v].iloc[:, 1:].to_numpy()),
                torch.Tensor(covs[covs.vowel == v].iloc[:, 1:].to_numpy()).view(len(dimensions), -1)
            )
        )

    samples = []
    labels = []
    freq_dist = torch.distributions.Categorical(torch.Tensor(counts.n.to_numpy()))

    for i in range(num_samples):
        cat = freq_dist.sample()
        val = dists[int(cat)].sample()
        samples.append(val[0])
        labels.append(counts.vowel[int(cat)])

    samples = torch.stack(samples)

    return samples, labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('mu_file', type=str)
    parser.add_argument('cov_file', type=str)
    parser.add_argument('counts_file', type=str)
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('dims', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    params = {
        # Means to use for in prior for each acoustic variable
        'prior_means': {
            'f1': 500,
            'f2': 1500,
            'duration': 275,
            'f0': 200,
            'f1-20-50': 0,
            'f2-20-50': 0,
            'f1-50-80': -20,
            'f2-50-80': 80,
            'f3': 2500,
            'f3-20-50': -10,
            'f3-50-80': -20,
            'f4': 4000
        },

        # Dimensions to use in simulation
        'dimensions': args.dims,

        # Alpha for dirichlet process
        'alpha': 10,

        # Annealing factor
        'annealing_factor': 9,

        'print_every': 10,

        'num_samples': 10000
    }

    file_bits = path.split(args.mu_file)[1].split('_')
    language = file_bits[0]
    register = file_bits[2]

    mu_full_path = path.join(args.input_folder, args.mu_file)
    cov_full_path = path.join(args.input_folder, args.cov_file)
    count_full_path = path.join(args.input_folder, args.counts_file)
    samples, labels = sample_inputs(
        mu_full_path, cov_full_path, count_full_path, params['dimensions']
    )

    learned_z, cats, lls = run(samples, params)

    output = pd.DataFrame(samples)
    output.columns = params['dimensions']

    output['vowel'] = labels
    output['learned_cat'] = learned_z.int()
    output.to_csv(
        path.join(args.output_folder, '{}_{}_{}.csv'.format(language, register, '_'.join(params['dimensions']))),
        index=False
    )

    cat_mus = pd.DataFrame(torch.stack(cats['cat_mus']))
    cat_mus['vowel'] = cat_mus.index
    cat_mus.columns = params['dimensions'] + ['vowel']
    cat_mus.to_csv(
        path.join(args.output_folder, '{}_{}_mus_{}.csv'.format(language, register, '_'.join(params['dimensions']))),
        index=False
    )

    cat_covs = torch.stack(cats['cat_covs'])
    cat_covs = pd.DataFrame(cat_covs.reshape(cat_covs.shape[0], -1))
    cat_covs['vowel'] = cat_covs.index
    cov_colnames = ['-'.join([dim1, dim2]) for dim1 in params['dimensions'] for dim2 in params['dimensions']] + ['vowel']
    cat_covs.columns = cov_colnames
    cat_covs.to_csv(
        path.join(args.output_folder, '{}_{}_covs_{}.csv'.format(language, register, '_'.join(params['dimensions']))), 
        index=False
    )

    lls = pd.DataFrame(lls)
    lls.columns = ['iteration', 'log_likelihood']
    lls.to_csv(
        path.join(args.output_folder, '{}_{}_ll_{}.csv'.format(language, register, '_'.join(params['dimensions']))), 
        index=False
    )
