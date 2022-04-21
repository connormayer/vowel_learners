import matplotlib.pyplot as plt
import torch

from input_sampler import sample_inputs
from math import log
from pprint import pprint
from torch.distributions import Categorical

torch.set_default_dtype(torch.float64)
# For debugging
# import random
# random.seed(0)
# torch.manual_seed(0)

#################
# SAMPLING CODE #
#################

def resample_z(z, x, anneal, phonemes, params):
    """
    Goes through each token, removes its category assignment, and samples a new
    one.
    """
    for token_idx in range(len(z)):
        remove_token(z, x, token_idx, phonemes)
        add_token(z, x, token_idx, anneal, phonemes, params)

def create_phoneme(phonemes, params):
    """
    Creates a new phoneme category with default values. We lay out the parameters
    of the categories in four separate lists rather than having a single object
    per category because this format simplifies the vector calculations.
    """
    phonemes['cat_mus'].append(params['mu_0'].clone())
    phonemes['cat_covs'].append(params['S_0'].clone())
    phonemes['cat_nus'].append(params['nu_0'])
    phonemes['cat_counts'].append(0)

def add_values(x, token_idx, cat, phonemes, params, data_ss=0, n=1):
    """
    Updates the parameters of a phoneme category after a token is added to it
    """
    mu = phonemes['cat_mus'][cat]
    cov = phonemes['cat_covs'][cat]
    nu = phonemes['cat_nus'][cat]
    
    cov += data_ss    
    mu_diff = x[token_idx] - mu

    cov += ((nu * n) / (nu + n)) * mu_diff * mu_diff[:, None]
    mu *= nu / (nu + n)
    scaled_x = x[token_idx] * (n / (nu + n))
    mu += scaled_x
    phonemes['cat_nus'][cat] += n

def subtract_values(x, token_idx, cat, phonemes, data_ss=0, n=1):
    """
    Updates the parameters of a phoneme category after a token is removed 
    from it
    """
    phonemes['cat_nus'][cat] -= n

    mu = phonemes['cat_mus'][cat]
    cov = phonemes['cat_covs'][cat]
    nu = phonemes['cat_nus'][cat]
    
    scaled_x = x[token_idx] * (n / (nu + n))
    mu -= scaled_x
    mu *= (nu + n) / nu

    mu_diff = x[token_idx] - mu
    cov -= data_ss
    cov -= (nu * n) / (nu + n) * mu_diff * mu_diff[:, None]

def add_token(z, x, token_idx, anneal, phonemes, params):
    """
    Samples a new phoneme category and adds a token to it
    """
    log_prior = get_prior(phonemes, params)
    log_likelihood = get_likelihood(x, token_idx, phonemes, params)

    log_unnorm_post = log_likelihood + log_prior

    # Apply annealing
    if anneal > 0:
        max_prob = torch.max(log_unnorm_post)
        log_unnorm_post = anneal * (log_unnorm_post - max_prob)

    # Sample a category based on probability
    post = torch.exp(log_unnorm_post - log_unnorm_post.logsumexp(-1, keepdim=True))
    new_cat = Categorical(post).sample()

    # Create a new category if necessary
    if new_cat > phonemes['max_cat']:
        phonemes['max_cat'] = new_cat
        create_phoneme(phonemes, params)

    # Update category parameters
    add_values(x, token_idx, new_cat, phonemes, params)
    # Update category assignments
    z[token_idx] = new_cat
    phonemes['cat_counts'][new_cat] += 1

def remove_token(z, x, token_idx, phonemes):
    """
    Removes a token from its phoneme category
    """
    # Remove category assignment for token
    prev_cat = int(z[token_idx])
    z[token_idx] = -1
    phonemes['cat_counts'][prev_cat] -= 1
    subtract_values(x, token_idx, prev_cat, phonemes)

    # Are there still instances of this category?
    if phonemes['cat_counts'][prev_cat] == 0:
        # Decrement all category labels above prev_cat
        z[z > prev_cat] -= 1
        # Delete phoneme from our phoneme lists
        phonemes['cat_mus'].pop(prev_cat)
        phonemes['cat_covs'].pop(prev_cat)
        phonemes['cat_nus'].pop(prev_cat)
        phonemes['cat_counts'].pop(prev_cat)
        phonemes['max_cat'] -= 1

def get_prior(phonemes, params):
    """
    Calculates (something proportional to) the prior over existing categories
    and a new category
    """
    pi = torch.Tensor(phonemes['cat_counts'] + [params['alpha']])
    return torch.log(pi)

def get_likelihood(x, token_idx, phonemes, params, data_ss=0, n=1):
    """
    Calculates the likelihood of a token being generated each existing category
    or a new category given the existing category assignments
    """
    dims = x.shape[1]
    mus = torch.stack(phonemes['cat_mus'] + [params['mu_0']])
    covs = torch.stack(phonemes['cat_covs'] + [params['S_0']])
    nus = torch.Tensor(phonemes['cat_nus'] + [params['nu_0']])

    S_c = covs + data_ss
    mu_diff = x[token_idx] - mus
    S_c += (n * nus[:, None, None]) / (n + nus[:, None, None]) * mu_diff[:, None, :] * mu_diff[:, :, None]

    p = 0

    nus_tensor = nus.repeat(dims, 1)
    t1 = (nus_tensor - params['dims_tensor'])
    t2 = t1 / 2
    t1 = (n + t1) / 2

    p += sum(torch.lgamma(t1) - torch.lgamma(t2))

    p += (nus / 2) * torch.logdet(covs)
    p -= (dims * n / 2) * log(torch.pi)
    p -= (dims / 2) * torch.log((nus + n) / nus)
    p -= ((nus + n) / 2) * torch.logdet(S_c)

    return p

def get_log_det(t):
    """
    This is a implementation of logdet that is a bit more efficient 
    than the torch.logdet function, but assumes matrices are
    symmetrical.
    """
    cholesky = torch.linalg.cholesky(t)
    diags = torch.diagonal(cholesky, dim1=-2, dim2=-1)
    return torch.sum(torch.log(diags), 1) * 2

def get_joint_probability(x, z, phonemes, params):
    """
    Computes the joint probability of the data and category assignments
    p(x,z|alpha,mu_0,nu_0,S_0) = p(w|z,mu_0,nu_0,S_0) p(z|alpha)

    This could be implemented more efficiently, but it doesn't run very often.
    """
    total_n = 0
    counts_tensor = torch.Tensor(phonemes['cat_counts'])

    # Compute p(z|alpha)
    # Compute numerator
    logprob = (phonemes['max_cat'].item() + 1) * log(params['alpha'])
    logprob += torch.sum(torch.lgamma(counts_tensor))
    total_n += torch.sum(counts_tensor)

    # Compute denominator
    logprob -= sum([log(i + params['alpha']) for i in range(int(total_n.item()))])

    # Compute p(w|z,mu_0,nu_0,S_0)
    # Reset category means, vars, and nus
    phonemes['cat_mus'] = [params['mu_0'].clone() for _ in range(phonemes['max_cat'] + 1)] 
    phonemes['cat_covs'] = [params['S_0'].clone() for _ in range(phonemes['max_cat'] + 1)] 
    phonemes['cat_nus'] = [params['nu_0']] * (phonemes['max_cat'] + 1)

    cat_mu_sums = torch.stack([torch.sum(x[z==k], 0) for k in range(phonemes['max_cat'] + 1)])

    cat_cov_sums = []
    for k in range(phonemes['max_cat'] + 1):
        cat_tokens = x[z==k]
        cat_cov_sum = torch.sum(
            torch.stack([
                token.view(-1, 1) @ token.view(1, -1)
                for token in cat_tokens
            ]), 
            0
        )
        cat_cov_sums.append(cat_cov_sum)
    cat_cov_sums = torch.stack(cat_cov_sums)

    n = torch.Tensor(phonemes['cat_counts'])

    # Correct mean and sum-of-squares terms
    # Normalize mean, E[X]
    cat_mu_sums = cat_mu_sums.div(n[:, None])
    # n * E[X^2] - n * E[X]^2
    cat_cov_sums += -n[:, None, None] * cat_mu_sums[:, None, :] * cat_mu_sums[:, :, None]

    logprob += sum([
        get_likelihood(
            cat_mu_sums, k, phonemes, params, 
            data_ss= torch.stack([cat_cov_sums[k]] * (phonemes['max_cat'] + 2)), 
            n=n[k])[k] 
        for k in range(phonemes['max_cat'] + 1)
    ])

    for k in range(phonemes['max_cat'] + 1):
        add_values(cat_mu_sums, k, k, phonemes, params, data_ss=cat_cov_sums[k], n=n[k])
    
    return logprob

def get_annealing_factor(iteration, params):
    if params['annealing_factor']:
        anneal = min(1, log(2 + iteration) / params['annealing_factor'])
    else:
        anneal = 0

    return anneal

def gibbs_sample(x, params, num_samples=10000, print_every=10000):
    """
    Kicks off the Gibbs sampling process
    """
    ##################
    # INITIALIZATION #
    ##################

    print("Initializing parameters")
    z = torch.zeros([x.shape[0]]) - 1
    anneal = get_annealing_factor(0, params)

    phonemes = {
        'cat_mus': [],
        'cat_covs': [],
        'cat_nus': [],
        'cat_counts': [],
        'max_cat': -1
    }

    # Initialize a few cached values
    params['dims_tensor'] = torch.arange(x.shape[1])[:, None]

    # First pass to initialize token categories
    for token_idx in range(len(z)):
        add_token(z, x, token_idx, anneal, phonemes, params)

    print("Beginning sampling...")

    for b in range(num_samples):
        print("Iteration {}".format(b))
        anneal = get_annealing_factor(b, params)

        resample_z(z, x, anneal, phonemes, params)

        # if b % print_every == 0:
        #     ll = get_joint_probability(x, z, phonemes, params)
        #     print(ll)
        #     pprint(phonemes)

    return z, phonemes

########################
# Data generation code #
########################

def generate_data(params, pi, mu, sigma):
    """
    Given category probabilities, means, and variances, samples a number of
    acoustic vowel tokens. This can be used to generate input for the 
    simulation.
    """
    z = torch.distributions.Categorical(pi).sample((N,)) + 1
    x = torch.distributions.Normal(mu[z - 1], sigma).sample()
    return z, x

######################
# Visualization code #
######################

def plot_data(z, x):
    """
    Creates a simple scatterplot of the datapoints and their category assignments
    """
    return plt.scatter(*x.T, c=z)

def plot_means(mu):
    """
    Creates a scatter plot of the category means
    """
    K = mu.shape[0]
    return plt.scatter(*mu.T, c=torch.arange(K))

if __name__ == "__main__":
    # A dictionary to hold the simulation parameters
    params = {
        #########################################
        # PARAMS RELATED TO SAMPLING INPUT DATA #
        #########################################
        # Number of word_samples
        'N': 5000,

        # Files for sampling input data
        'vowel_file': 'corpus_data/hillenbrand_vowel_acoustics.csv',
        'word_file': 'corpus_data/childes_counts_prons.csv',

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
            'f3': 2800,
            'f3-20-50': -10,
            'f3-50-80': -20,
            'f4': 4000
        },

        # Acoustic variables to use in simulation
        'dimensions': [
            # 'f0',
            'f1',
            'f2',
            # 'f3',
            # 'f4',
            # 'duration',
            # 'f1-20-50',
            # 'f2-20-50',
            # 'f3-20-50',
            # 'f3-50-80',
            # 'f1-50-80',
            # 'f2-50-80'
        ],

        # Alpha for dirichlet process
        'alpha': 10,

        # Annealing factor
        'annealing_factor': 9,

        # Default parameters for category distributions
        #'mu_0': torch.tensor([500., 1500.]),
        'S_0': torch.eye(2),
        'nu_0': 1.001
    }

    # Set mu_0 to contain only dimensions we're using
    params['mu_0'] = torch.Tensor([
        params['prior_means'][dim] for dim in params['dimensions']
    ])

    words, vowel_samples, vowel_labels = sample_inputs(
        params['vowel_file'], params['word_file'], params['dimensions']
    )

    learned_z, cats = gibbs_sample(
        torch.stack(vowel_samples), params, print_every=1000, num_samples=10000
    )
    # TODO WRITE OUTPUT
    print(cats)


# # Generate data
# # The probability of observing each category
# true_pi = torch.Tensor([0.33, 0.33, 0.34])

# # The means of each category
# true_mu = torch.Tensor([
#         [300, 2500],
#         [1000, 1000],
#         [300, 700]
#     ])

# # The standard deviation of each category
# true_sigma = 100
# true_z, data_x = generate_data(params, true_pi, true_mu, true_sigma)
# # data_x = torch.Tensor([
# #     [400, 1600],
# #     [600, 1400]
# # ])