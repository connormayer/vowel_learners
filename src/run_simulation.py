from distributional_learner import run
from os import path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def sample_inputs(mu_file, cov_file, counts_file, dimensions, num_samples):
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
        mu = mus[mus.vowel == v].iloc[:, 1:].to_numpy()[0]
        cov = covs[covs.vowel == v].iloc[:, 1:].to_numpy()[0]
        cov = cov.reshape(len(dimensions), -1)
        dists.append([mu, cov])

    samples = []
    labels = []

    freq_dist = torch.distributions.Categorical(torch.Tensor(counts.n.to_numpy()))

    for i in range(num_samples):
        cat = freq_dist.sample()
        mu, cov = dists[int(cat)]
        val = np.random.multivariate_normal(mu, cov)
        samples.append(val)
        labels.append(counts.vowel[int(cat)])

    samples = np.stack(samples)

    return samples, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mu_file', type=str, 
        help='The file containing category means'
    )
    parser.add_argument(
        'cov_file', type=str, 
        help='The file containing category covariance matrices'
    )
    parser.add_argument(
        'counts_file', type=str, 
        help='The file containing category counts'
    )
    parser.add_argument(
        'input_folder', type=str, 
        help='The folder containing the input files'
    )
    parser.add_argument(
        'output_folder', type=str, 
        help='The folder where the output will be saved'
    )
    parser.add_argument(
        '--suffix', default='', type=str, 
        help="A string to suffix onto output filenames. Defaults to the empty string."
    )
    parser.add_argument(
        '--vowel_samples', default=5000, type=int,
        help="The number of vowel samples to use as input to the learner. Defaults to 5000."
    )
    parser.add_argument(
        '--iterations', default=10000, type=int,
        help="The number of iterations the learner should use. Defaults to 10,000."
    )
    parser.add_argument(
        '--barks', action='store_true', default=False,
        help="If this flag is provided, category priors will be in Barks rather "
             "than Hz. Defaults to using Hz."
    )
    parser.add_argument(
        '--alpha', default=10, type=float,
        help="The value to use for the alpha parameter, which controls how "
             "likely the learner is to create a new category. Defaults to 10."
    )
    parser.add_argument(
        '--annealing_factor', default=9, type=float,
        help="The annealing factor. Defaults to 9."
    )
    parser.add_argument(
        '--print_every', default=100, type=int,
        help="After how many iterations a status message should be printed. Defaults to 100."
    )
    parser.add_argument(
        '--dims', nargs=argparse.REMAINDER, default=['f1', 'f2'],
        help="The dimensions to use. These names must match the column names in "
             "the input files. This should be passed in as a space-separated "
             "list. This must also be the final argument. Defaults to 'f1 f2'."
    )
    args = parser.parse_args()

    params = {
        # Dimensions to use in simulation
        'dimensions': args.dims,

        # Alpha for dirichlet process
        'alpha': args.alpha,

        # Annealing factor
        'annealing_factor': args.annealing_factor,

        'print_every': args.print_every,

        'num_samples': args.iterations
    }

    # Means to use in prior for each acoustic variable
    if args.barks:
        params['prior_means'] = {
            'f1': 4.9191869918699185,
            'f2': 11.092832369942197,
            'duration': 275,
            'df1_on': -0.53,
            'df2_on': -0.53,
            'df1_off': -0.53,
            'df2_off': -0.53,
            'f3': 14.498026905829597,
            'df3_on': -0.53,
            'df3_off': -0.53,
            'f4': 17.463288590604026
        }
    else:
        params['prior_means'] = {
            'f1': 500,
            'f2': 1500,
            'duration': 275,
            'df1_on': 0,
            'df2_on': 0,
            'df1_off': 0,
            'df2_off': 0,
            'f3': 2500,
            'df3_on': 0,
            'df3_off': 0,
            'f4': 4000
        }

    file_bits = path.split(args.mu_file)[1].split('_')
    language = file_bits[0]
    register = file_bits[2]

    mu_full_path = path.join(args.input_folder, args.mu_file)
    cov_full_path = path.join(args.input_folder, args.cov_file)
    count_full_path = path.join(args.input_folder, args.counts_file)
    samples, labels = sample_inputs(
        mu_full_path, cov_full_path, count_full_path, params['dimensions'],
        num_samples=args.vowel_samples
    )

    learned_z, cats, lls = run(torch.Tensor(samples), params)

    output = pd.DataFrame(samples)
    output.columns = params['dimensions']

    output['vowel'] = labels
    output['learned_cat'] = learned_z
    output.to_csv(
        path.join(
            args.output_folder, 
            '{}_{}_{}_{}.csv'.format(language, register, '_'.join(params['dimensions']), args.suffix)
        ),
        index=False
    )

    cat_mus = pd.DataFrame(np.stack(cats['cat_mus']))
    cat_mus['vowel'] = cat_mus.index
    cat_mus.columns = params['dimensions'] + ['vowel']
    cat_mus.to_csv(
        path.join(
            args.output_folder, 
            '{}_{}_mus_{}_{}.csv'.format(language, register, '_'.join(params['dimensions']), args.suffix)
        ),
        index=False
    )

    cat_covs = np.stack(cats['cat_covs'])
    cat_covs = pd.DataFrame(cat_covs.reshape(cat_covs.shape[0], -1))
    cat_covs['vowel'] = cat_covs.index
    cov_colnames = [
        '-'.join([dim1, dim2]) for dim1 in params['dimensions'] 
        for dim2 in params['dimensions']
    ] + ['vowel']
    cat_covs.columns = cov_colnames
    cat_covs.to_csv(
        path.join(
            args.output_folder, 
            '{}_{}_covs_{}_{}.csv'.format(language, register, '_'.join(params['dimensions']), args.suffix)
        ), 
        index=False
    )

    lls = pd.DataFrame(lls)
    lls.columns = ['iteration', 'log_likelihood']
    lls.to_csv(
        path.join(
            args.output_folder, 
            '{}_{}_ll_{}_{}.csv'.format(language, register, '_'.join(params['dimensions']), args.suffix)
        ), 
        index=False
    )
