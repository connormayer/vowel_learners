import distributional_learner
import torch

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

# A dictionary to hold the simulation parameters
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
    # Alpha for dirichlet process
    'alpha': 10,

    # Annealing factor
    'annealing_factor': 9,

    'print_every': 1,

    'num_samples': 10000
}
params['dimensions'] = [
    # 'f0',
    'f1',
    'f2',
    'f3',
    # 'f4',
    # 'duration',
    # 'f1-20-50',
    # 'f2-20-50',
    # 'f3-20-50',
    # 'f3-50-80',
    # 'f1-50-80',
    # 'f2-50-80'
]


num_word_samples = 10000
vowel_file = 'corpus_data/hillenbrand_vowel_acoustics.csv'
word_file = 'corpus_data/childes_counts_prons.csv'

# words, vowel_samples, vowel_labels = sample_inputs(
#     vowel_file, word_file, params['dimensions'],
#     num_samples=10000
# )

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

vowel_samples = [
    torch.Tensor([100, 400, 2000]),
    torch.Tensor([160, 600, 2100])
]


learned_z, cats = distributional_learner.run(
    torch.stack(vowel_samples), params,
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