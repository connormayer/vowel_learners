import numpy as np
import os
import torch

root_dir = "/home/connor/ling/vowel_learning_project/Code/vowel_learners/outputs/spanish_ads_cds"
cds_mu_file = os.path.join(root_dir, 'cds_f1_f2_with_outliers/cds_cat_mus_f1_f2.pt')
cds_covs_file = os.path.join(root_dir, 'cds_f1_f2_with_outliers/cds_cat_covs_f1_f2.pt')

cds_mus = torch.load(cds_mu_file)
cds_covs = torch.load(cds_covs_file)

np.savetxt(
    os.path.join(root_dir, 'cds_f1_f2_with_outliers/cds_mus.csv'), 
    cds_mus, 
    delimiter=','
)
np.savetxt(
    os.path.join(root_dir, 'cds_f1_f2_with_outliers/cds_covs.csv'), 
    cds_covs.reshape(cds_covs.shape[0], -1),
    delimiter=','
)

ads_mu_file = os.path.join(root_dir, 'ads_cat_mus_f1_f2.pt')
ads_covs_file = os.path.join(root_dir, 'ads_cat_covs_f1_f2.pt')

ads_mus = torch.load(ads_mu_file)
ads_covs = torch.load(ads_covs_file)

np.savetxt(
    os.path.join(root_dir, 'ads_mus.csv'), 
    ads_mus, 
    delimiter=','
)
np.savetxt(
    os.path.join(root_dir, 'ads_covs.csv'), 
    ads_covs.reshape(ads_covs.shape[0], -1),
    delimiter=','
)