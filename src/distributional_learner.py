import numpy as np
import torch 

from scipy.special import gammaln, logsumexp

# For debugging
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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
    phonemes['cat_mus'].append(params['mu_0'].copy())
    phonemes['cat_covs'].append(params['S_0'].copy())
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
    mu = mu * (nu / (nu + n))
    scaled_x = x[token_idx] * (n / (nu + n))
    mu += scaled_x
    phonemes['cat_nus'][cat] += n

    # Force matrix to be symmetrical
    cov = make_symmetric(cov)

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
    mu = mu - scaled_x
    mu = mu * ((nu + n) / nu)

    mu_diff = x[token_idx] - mu
    cov -= data_ss
    cov -= (nu * n) / (nu + n) * mu_diff * mu_diff[:, None]

    # Force matrix to be symmetrical
    cov = make_symmetric(cov)

def make_symmetric(mat):
    return np.maximum(mat, mat.transpose())

def add_token(z, x, token_idx, anneal, phonemes, params):
    """
    Samples a new phoneme category and adds a token to it
    """
    log_prior = get_prior(phonemes, params)
    log_likelihood = get_likelihood(x, token_idx, phonemes, params)

    log_unnorm_post = log_likelihood + log_prior

    # Apply annealing
    if anneal > 0:
        max_prob = np.max(log_unnorm_post)
        log_unnorm_post = anneal * (log_unnorm_post - max_prob)

    # Sample a category based on probability
    post = np.exp(log_unnorm_post - logsumexp(log_unnorm_post))
    new_cat = np.random.choice(len(post), p=post)

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
    pi = np.array(phonemes['cat_counts'] + [params['alpha']])
    return np.log(pi)

def get_likelihood(x, token_idx, phonemes, params, data_ss=0, n=1):
    """
    Calculates the likelihood of a token being generated each existing category
    or a new category given the existing category assignments
    """
    dims = x.shape[1]
    mus = np.stack(phonemes['cat_mus'] + [params['mu_0']])
    covs = np.stack(phonemes['cat_covs'] + [params['S_0']])
    nus = np.array(phonemes['cat_nus'] + [params['nu_0']])

    S_c = covs + data_ss
    mu_diff = x[token_idx] - mus
    S_c += (n * nus[:, None, None]) / (n + nus[:, None, None]) * mu_diff[:, None, :] * mu_diff[:, :, None]

    p = 0

    nus_tensor = np.tile(nus, (dims, 1))
    t1 = (nus_tensor - params['dims_tensor'])
    t2 = t1 / 2
    t1 = (n + t1) / 2

    p += sum(gammaln(t1) - gammaln(t2))
    p += (nus / 2) * np.linalg.slogdet(covs)[1]
    p -= (dims * n / 2) * np.log(np.pi)
    p -= (dims / 2) * np.log((nus + n) / nus)
    p -= ((nus + n) / 2) * np.linalg.slogdet(S_c)[1]
    return p

def get_joint_probability(x, z, phonemes, params):
    """
    Computes the joint probability of the data and category assignments
    p(x,z|alpha,mu_0,nu_0,S_0) = p(w|z,mu_0,nu_0,S_0) p(z|alpha)

    This could be implemented more efficiently, but it doesn't run very often.
    """
    total_n = 0
    counts_tensor = np.array(phonemes['cat_counts'])

    # Compute p(z|alpha)
    # Compute numerator
    logprob = (phonemes['max_cat'] + 1) * np.log(params['alpha'])
    logprob += np.sum(gammaln(counts_tensor))
    total_n += np.sum(counts_tensor)

    # Compute denominator
    logprob -= sum([np.log(i + params['alpha']) for i in range(total_n)])

    # Compute p(w|z,mu_0,nu_0,S_0)
    # Reset category means, vars, and nus
    phonemes['cat_mus'] = [params['mu_0'].copy() for _ in range(phonemes['max_cat'] + 1)] 
    phonemes['cat_covs'] = [params['S_0'].copy() for _ in range(phonemes['max_cat'] + 1)] 
    phonemes['cat_nus'] = [params['nu_0']] * (phonemes['max_cat'] + 1)

    cat_mu_sums = np.stack([np.sum(x[z==k], 0) for k in range(phonemes['max_cat'] + 1)])

    cat_cov_sums = []
    for k in range(phonemes['max_cat'] + 1):
        cat_tokens = x[z==k]
        cat_cov_sum = np.sum(
            np.stack([
                token.reshape(-1, 1) @ token.reshape(1, -1)
                for token in cat_tokens
            ]), 
            0
        )
        cat_cov_sums.append(cat_cov_sum)
    cat_cov_sums = np.stack(cat_cov_sums)

    n = np.array(phonemes['cat_counts'])

    # Correct mean and sum-of-squares terms
    # Normalize mean, E[X]
    cat_mu_sums = np.divide(cat_mu_sums, n[:, None])
    # n * E[X^2] - n * E[X]^2
    cat_cov_sums += -n[:, None, None] * cat_mu_sums[:, None, :] * cat_mu_sums[:, :, None]

    logprob += sum([
        get_likelihood(
            cat_mu_sums, k, phonemes, params, 
            data_ss= np.stack([cat_cov_sums[k]] * (phonemes['max_cat'] + 2)), 
            n=n[k])[k] 
        for k in range(phonemes['max_cat'] + 1)
    ])

    for k in range(phonemes['max_cat'] + 1):
        add_values(cat_mu_sums, k, k, phonemes, params, data_ss=cat_cov_sums[k], n=n[k])
    
    return logprob

def get_annealing_factor(iteration, params):
    if params['annealing_factor']:
        anneal = min(1, np.log(2 + iteration) / params['annealing_factor'])
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
    z = np.zeros([x.shape[0]]) - 1
    anneal = get_annealing_factor(0, params)
    log_likelihoods = []

    phonemes = {
        'cat_mus': [],
        'cat_covs': [],
        'cat_nus': [],
        'cat_counts': [],
        'max_cat': -1
    }

    # Initialize a few cached values
    params['dims_tensor'] = np.arange(x.shape[1])[:, None]
    # First pass to initialize token categories

    for token_idx in range(len(z)):
        add_token(z, x, token_idx, anneal, phonemes, params)

    print("Beginning sampling...")

    for b in range(num_samples):
        print("Iteration {}".format(b))
        anneal = get_annealing_factor(b, params)

        resample_z(z, x, anneal, phonemes, params)

        if b % print_every == 0:
            ll = get_joint_probability(x, z, phonemes, params)
            log_likelihoods.append((b, ll.item()))
            print("Log likelihood: {}".format(ll))
            print("Num cats: {}".format(phonemes['max_cat'] + 1))

    return z, phonemes, log_likelihoods

#######
# RUN #
#######

def run(vowel_samples, params):
    # Set parameters for prior distributions based on dimensionality
    params['mu_0'] = np.array([
        params['prior_means'][dim] for dim in params['dimensions']
    ])
    params['S_0'] = np.eye(len(params['dimensions']))
    params['nu_0'] = len(params['dimensions']) - 1 + 0.001

    # Do Gibbs sampling
    learned_z, cats, log_likelihoods = gibbs_sample(
        vowel_samples, params, print_every = params['print_every'], 
        num_samples=params['num_samples']
    )

    return learned_z, cats, log_likelihoods
