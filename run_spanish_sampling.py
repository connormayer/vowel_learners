from distributional_learner import run
from os import path

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch

def sample_inputs(vowel_file, dimensions, num_samples=5000):
    vowel_acoustics = pd.read_csv(vowel_file)
    counts = vowel_acoustics[['vowel', 'n']]

    means = vowel_acoustics.filter(regex='(^mean_)|^vowel')
    sds = vowel_acoustics.filter(regex='(^sd_)|^vowel')

    mean_names = ['vowel'] + ['mean_{}'.format(dim) for dim in dimensions]
    sd_names = ['vowel'] + ['sd_{}'.format(dim) for dim in dimensions]

    means = means.filter(mean_names)
    sds = sds.filter(sd_names)

    dists = []

    for v in counts.vowel:
        dists.append(
            torch.distributions.Normal(
                torch.Tensor(means[means.vowel == v].iloc[:, 1:].to_numpy()),
                torch.Tensor(sds[sds.vowel == v].iloc[:, 1:].to_numpy())
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
    parser.add_argument('train_file', type=str)
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
        # [
        #     # 'f0',
        #     'f1',
        #     'f2',
        #     #'f3',
        #     # 'f4',
        #     #'duration',
        #     # 'f1-20-50',
        #     # 'f2-20-50',
        #     # 'f3-20-50',
        #     # 'f3-50-80',
        #     # 'f1-50-80',
        #     # 'f2-50-80'
        # ],

        # Alpha for dirichlet process
        'alpha': 10,

        # Annealing factor
        'annealing_factor': 9,

        'print_every': 10,

        'num_samples': 10000
    }

    train_file = args.train_file
    #train_file = 'corpus_data/distributions/spanish_summary_train.csv'
    language = path.split(train_file)[1].split('_')[0]

    samples, labels = sample_inputs(train_file, params['dimensions'])

    learned_z, cats, lls = run(samples, params)

    output = pd.DataFrame(samples)
    output.columns = params['dimensions']

    output['vowel'] = labels
    output['learned_cat'] = learned_z.int()
    output.to_csv('outputs/{}_{}.csv'.format(language, '_'.join(params['dimensions'])))

    cat_mus = torch.stack(cats['cat_mus'])
    torch.save(cat_mus, 'outputs/{}_cat_mus_{}.pt'.format(language, '_'.join(params['dimensions'])))
    cat_covs = torch.stack(cats['cat_covs'])
    torch.save(cat_covs, 'outputs/{}_cat_covs_{}.pt'.format(language, '_'.join(params['dimensions'])))

    lls = pd.DataFrame(lls)
    lls.columns = ['iteration', 'log_likelihood']
    lls.to_csv('outputs/{}_ll_{}.csv'.format(language, '_'.join(params['dimensions'])), index=False)
