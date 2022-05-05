from distributional_learner import run

import pandas as pd
import torch

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
    'dimensions': {
        # 'f0',
        'f1',
        'f2',
        'f3',
        # 'f4',
        'duration',
        # 'f1-20-50',
        # 'f2-20-50',
        # 'f3-20-50',
        # 'f3-50-80',
        # 'f1-50-80',
        # 'f2-50-80'
    },

    # Alpha for dirichlet process
    'alpha': 10,

    # Annealing factor
    'annealing_factor': 9,

    'print_every': 10,

    'num_samples': 10000
}

ads = pd.read_csv('corpus_data/spanish_ads_no_outliers.csv')
ids = pd.read_csv('corpus_data/spanish_cds_no_outliers.csv')

ads_vowel_samples = ads[[
    'f1', 'f2', 'f3', 'duration', 'df1_on', 'df2_on', 'df3_on',
    'df1_off', 'df2_off', 'df3_off'
]]

ids_vowel_samples = ids[[
    'f1', 'f2', 'f3', 'duration', 'df1_on', 'df2_on', 'df3_on',
    'df1_off', 'df2_off', 'df3_off'
]]

# # RUN ADS
# print("Running on ADS")
# ads_learned_z, ads_cats = gibbs_sample(
#     torch.tensor(ads_vowel_samples.values[:, :4]), params, print_every=1, num_samples=10000
# )
# ads['learned_cats'] = ads_learned_z.int()
# ads.to_csv('outputs/spanish_ads_cds/ads_results_f1_f2_f3_dur.csv')

# ads_cat_mus = torch.stack(ads_cats['cat_mus'])
# torch.save(ads_cat_mus, 'outputs/spanish_ads_cds/ads_cat_mus_f1_f2_f3_dur.pt')
# ads_cat_covs = torch.stack(ads_cats['cat_covs'])
# torch.save(ads_cat_covs, 'outputs/spanish_ads_cds/ads_cat_covs_f1_f2_f3_dur.pt')

# RUN CDS 
print("Running on CDS")
ids_learned_z, ids_cats, ids_lls = run(
    torch.Tensor(ids_vowel_samples.values[:, :len(params['dimensions'])]), params
)
ids['learned_cats'] = ids_learned_z.int()
ids.to_csv('outputs/spanish_ads_cds/cds_results_f1_f2_f3_dur.csv')

ids_cat_mus = torch.stack(ids_cats['cat_mus'])
torch.save(ids_cat_mus, 'outputs/spanish_ads_cds/cds_cat_mus_f1_f2_f3_dur.pt')
ids_cat_covs = torch.stack(ids_cats['cat_covs'])
torch.save(ids_cat_covs, 'outputs/spanish_ads_cds/cds_cat_covs_f1_f2_f3_dur.pt')

ids_lls = pd.DataFrame(ids_lls)
ids_lls.columns = ['iteration', 'log_likelihood']
ids_lls.to_csv('outputs/spanish_ads_cds/cds_ll_f1_f2_f3_dur.csv', index=False)

# import pdb; pdb.set_trace()