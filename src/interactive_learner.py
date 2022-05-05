import matplotlib.pyplot as plt
import torch

from collections import defaultdict
# TODO: Factor into helper file
from copy import deepcopy
from distributional_learner import add_token, add_values, create_phoneme, get_likelihood, get_prior, subtract_values
from input_sampler import sample_inputs
from math import log, inf
from pprint import pprint
from torch.distributions import Categorical

torch.set_default_dtype(torch.float64)
# For debugging
import random
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

#################
# SAMPLING CODE #
#################

def get_annealing_factor(iteration, params):
    if params['annealing_factor']:
        anneal = min(1, log(2 + iteration) / params['annealing_factor'])
    else:
        anneal = 0

    return anneal

def create_lexical_item(word, phoneme_idxs, lexical_items, params):
    lexical_items.append({
            'phoneme_sums': torch.zeros(word['vowel_len'], 
                                        len(params['dimensions'])),
            'phoneme_ss': torch.zeros(word['vowel_len'], 
                                      len(params['dimensions']),
                                      len(params['dimensions'])),
            'count': 0,
            'template': word['template'],
            'cons': word['cons'],
            'len': word['len'],
            'vowel_len': word['vowel_len'],
            'cons_len': word['cons_len'],
            'phoneme_idxs': torch.IntTensor(phoneme_idxs)
    })

def destroy_lexical_item(lex_item, lex_idx, ncons, vow_cats, lex_cats, 
                         lexical_items, phonemes):
    lexical_items.pop(lex_idx)
    lex_cats[lex_cats > lex_idx] -= 1

    for phon_idx in lex_item['phoneme_idxs']:
        phonemes['cat_counts'][phon_idx] -= 1
        if phonemes['cat_counts'][phon_idx] == 0:
            destroy_phoneme(phon_idx, lexical_items, vow_cats, phonemes)
            lex_item['phoneme_idxs'][lex_item['phoneme_idxs'] > phon_idx] -= 1
                
    for con in lex_item['cons']:
        ncons[con] -= 1

def add_phoneme_tokens(lex, phonemes, params):
    for phon_cat in lex['phoneme_idxs']:
        if phon_cat > phonemes['max_cat']:
            phonemes['max_cat'] = int(phon_cat)
            create_phoneme(phonemes, params)
        phonemes['cat_counts'][phon_cat] += 1

def sample_lexical_items(word, phonemes, anneal, params, old_cat=None):
    """
    pi: Phoneme counts, proportional to prior for phoneme category
    word: Target word
    """
    sampled_words = []
    probs_orig = torch.Tensor(phonemes['cat_counts'] + [params['alpha']])
    diff = 0

    if old_cat:
        sampled_words.append(old_cat)
        diff = 1

    for i in range(params['num_lexical_samples'] - diff):
        probs = probs_orig.clone()
        log_probs = torch.log(probs)
        max_prob = torch.max(log_probs)
        annealed_probs = torch.exp(anneal * (log_probs - max_prob))

        new_cat_idx = len(annealed_probs) - 1

        sampled_vowels = []

        probs = torch.concat((probs, torch.zeros(len(word['vowel_idxs']))))
        annealed_probs = torch.concat((annealed_probs, torch.zeros(len(word['vowel_idxs']))))

        for vowel_idx in word['vowel_idxs']:
            sampled_cat = torch.distributions.Categorical(annealed_probs).sample()

            if sampled_cat == new_cat_idx:
                # New phoneme
                sampled_vowels.append(sampled_cat)
                if (sampled_cat + 1) < len(probs):
                    probs[sampled_cat + 1] = params['alpha']
                    annealed_probs[sampled_cat + 1] = torch.exp(anneal * (torch.log(probs[sampled_cat]) - max_prob))
                probs[sampled_cat] = 1
                annealed_probs[sampled_cat] = torch.exp(anneal * (0 - max_prob))
                new_cat_idx += 1
            else:
                # Existing phoneme
                sampled_vowels.append(sampled_cat)
                probs[sampled_cat] += 1
                annealed_probs[sampled_cat] = torch.exp(
                    anneal * (torch.log(probs[sampled_cat]) - max_prob)
                )

        create_lexical_item(word, sampled_vowels, sampled_words, params)

    return sampled_words

def add_word(w_idx, word, ncons, anneal, vowels, phonemes, lexical_items, 
             vow_cats, lex_cats, params, old_cat=None):
    # Sample pseudo-words
    pseudo_words = sample_lexical_items(word, phonemes, anneal, params, old_cat=old_cat)

    # Get prior
    log_eff_beta = get_word_prior(word, pseudo_words, ncons, anneal, phonemes, params)

    # Compute log likelihood of newly sampled tables
    pseudo_cats_probs = get_word_posterior(word, pseudo_words, vowels, phonemes, log_eff_beta)
    lex_cats_probs = get_word_posterior(word, lexical_items, vowels, phonemes)

    # Sample a new category
    probs = torch.cat((lex_cats_probs, pseudo_cats_probs))
    max_prob = torch.max(probs)
    probs = torch.exp(anneal * (probs - max_prob))

    new_cat = Categorical(probs).sample()

    if new_cat >= len(lexical_items):
        new_lex = pseudo_words[new_cat - len(lexical_items)]
        for con in new_lex['cons']:
            ncons[con] += 1
        lexical_items.append(new_lex)
        try:
            add_phoneme_tokens(new_lex, phonemes, params)
        except:
            import pdb; pdb.set_trace()
        new_cat = len(lexical_items) - 1
    else:
        new_lex = lexical_items[new_cat]

    lex_cats[w_idx] = new_cat
    new_lex['count'] += 1

    # Update parameters of phonemes in word
    for i, idx in enumerate(word['vowel_idxs']):
        add_values(
            vowels, idx, new_lex['phoneme_idxs'][i], phonemes, 
            params
        )
        vow_cats[idx] = new_lex['phoneme_idxs'][i]
        new_lex['phoneme_sums'][i] += vowels[idx]
        new_lex['phoneme_ss'][i] += vowels[idx].view(-1, 1) @ vowels[idx].view(1, -1)

def remove_word(w, word, ncons, vowels, phonemes, lexical_items, vow_cats, lex_cats):
    old_cat = int(lex_cats[w])
    lex_item = lexical_items[old_cat]
    lex_item['count'] -= 1
    lex_cats[w] = -1

    for v, vowel_idx in enumerate(word['vowel_idxs']):
        subtract_values(vowels, vowel_idx, int(vow_cats[vowel_idx]), phonemes)
        lex_item['phoneme_sums'][v] -= vowels[vowel_idx]
        lex_item['phoneme_ss'][v] -= vowels[vowel_idx].view(-1, 1) @ vowels[vowel_idx].view(1, -1)

    # If lexical category no longer used, delete table but keep as part of sample from prior
    word_copy = None
    if lex_item['count'] == 0:
        word_copy = deepcopy(lex_item)
        destroy_lexical_item(
            lex_item, old_cat, ncons, vow_cats, lex_cats, lexical_items, phonemes
        )

    return word_copy

def destroy_phoneme(phon_idx, lexical_items, vow_cats, phonemes):
    # Decrement all category labels above prev_cat
    vow_cats[vow_cats > phon_idx] -= 1
    for l in lexical_items:
        phons = l['phoneme_idxs']
        phons[phons > phon_idx] -= 1

    # Delete phoneme from our phoneme lists
    phonemes['cat_mus'].pop(phon_idx)
    phonemes['cat_covs'].pop(phon_idx)
    phonemes['cat_nus'].pop(phon_idx)
    phonemes['cat_counts'].pop(phon_idx)
    phonemes['max_cat'] -= 1

def get_word_prior(word, pseudo_words, ncons, anneal, phonemes, params):
    # Figure out effective value of beta for this lexical frame
    # Divide by number of samples from the prior
    log_eff_beta = log(params['beta']) - log(params['num_lexical_samples'])
    # Term for length: g^(length - 1) * (1 - g)
    log_eff_beta += (word['len'] - 1) * log(params['geometric'])
    log_eff_beta += log(1 - params['geometric'])

    # Term for syllable template: p_cons ^ num_cons * (1 - p_cons) ^ num_vowels
    log_eff_beta += word['cons_len'] * log(params['cons'])
    log_eff_beta += word['vowel_len'] * log(1 - params['cons'])

    my_ncons = defaultdict(int)
    total_ncons = sum(ncons.values())

    for j, con in enumerate(word['cons']):
        my_ncons[con] += 1
        # Divide by the denominator total_ncons * total_ncons + 1 * ... * total_ncons + cons_length - 1
        if (total_ncons + j) > 0:
            log_eff_beta -= log(total_ncons + j)
        else:
            log_eff_beta -= log(params['alpha'])

    for c, count in my_ncons.items():
        # numerator is ncons[c] * ncons[c] + 1 * ... * (ncons[c] + my_ncons[c] - 1)
        if count > 0:
            if ncons[c] > 0:
                log_eff_beta += log(ncons[c])
            else:
                log_eff_beta += log(params['alpha'])
            for k in range(1, count):
                log_eff_beta += log(ncons[c] + k)

    return log_eff_beta

def get_word_posterior(word, candidates, vowels, phonemes, prior=None):
    probs = torch.zeros(len(candidates))

    for i, lex_cand in enumerate(candidates):
        if lex_cand['template'] == word['template']:
            duplicate = torch.zeros([word['vowel_len']])

            for j, vowel in enumerate(word['vowel_idxs']):
                if not duplicate[j]:
                    acoust = vowels[word['vowel_idxs'][j]]
                    ss = acoust[:, None] * acoust

                    count = 1
                    for l in range(j + 1, word['vowel_len']):
                        if lex_cand['phoneme_idxs'][l] == lex_cand['phoneme_idxs'][j]:
                            duplicate[l] = 1
                            val = vowels[word['vowel_idxs'][l]]
                            acoust = acoust + val
                            ss += val[:, None] * val
                            count += 1

                    acoust /= count
                    ss -= count * acoust[:, None] * acoust
                    # TODO: This is weird
                    acoust_prob = get_likelihood(
                        acoust.view(1, -1), 0, phonemes, params, data_ss=ss, n=count,
                    )
                    if lex_cand['phoneme_idxs'][j] > phonemes['max_cat']:
                        index = -1
                    else:
                        index = lex_cand['phoneme_idxs'][j]

                    probs[i] += acoust_prob[index]

                    if prior:
                        probs[i] += prior
                    else:
                        probs[i] += log(lex_cand['count'])
        else:
            probs[i] = -inf

    return probs

def resample_words(words, ncons, anneal, vowels, phonemes, lexical_items, vow_cats, lex_cats):
    for w, word in enumerate(words):
        old_cat = remove_word(
            w, word, ncons, vowels, phonemes, lexical_items, vow_cats, lex_cats
        )
        add_word(
            w, word, ncons, anneal, vowels, phonemes, lexical_items, vow_cats,
            lex_cats, params, old_cat=old_cat
        )

def remove_vowel(j, phon_idx, lex_idx, lex_item, lexical_items, words, phonemes,
                 vow_cats, lex_cats):
    phon_idx = int(phon_idx)
    # Remove vowel
    # Correct mean and sum-of-squares terms
    lex_item['phoneme_idxs'][j] = -1
    count = lex_item['count']
    phoneme_mean = (1 / count) * lex_item['phoneme_sums'][j]
    ss_change = (1 / count) * phoneme_mean.view(-1, 1) @ phoneme_mean.view(1, -1)
    phoneme_ss = lex_item['phoneme_ss'][j] - ss_change

    # Decrement count of current phoneme
    phonemes['cat_counts'][phon_idx] -= 1
    subtract_values(
        phoneme_mean.view(1, -1), 0, phon_idx, phonemes, phoneme_ss,
        lex_item['count']
    )
    # Decategorize vowels from words belonging to this lexical item
    for k, cat in enumerate(lex_cats):
        if cat == lex_idx:
            vow_idx = words[k]['vowel_idxs'][j]
            vow_cats[vow_idx] = -1

    if phonemes['cat_counts'][phon_idx] == 0:
        destroy_phoneme(phon_idx, lexical_items, vow_cats, phonemes)

    return phoneme_mean, phoneme_ss

def add_vowel(j, phoneme_mean, phoneme_ss, i, lex_item, vow_cats, lex_cats, 
              phonemes, anneal, params):
    # Get log likelihood of phonemes
    probs = get_likelihood(
        phoneme_mean.view(1, -1), 0,  phonemes, params, 
        data_ss=phoneme_ss, n=lex_item['count']
    )
    probs += get_prior(phonemes, params)

    if anneal > 0:
        max_prob = torch.max(probs)
        probs = anneal * (probs - max_prob)
    post = torch.exp(probs - probs.logsumexp(-1, keepdim=True))

    new_cat = Categorical(post).sample()

    if new_cat > phonemes['max_cat']:
        phonemes['max_cat'] = new_cat
        create_phoneme(phonemes, params)

    add_values(
        phoneme_mean.view(1, -1), 0, new_cat, phonemes, params, phoneme_ss,
        lex_item['count']
    )
    phonemes['cat_counts'][new_cat] += 1
    lex_item['phoneme_idxs'][j] = new_cat

    for k, cat in enumerate(lex_cats):
        if cat == i:
            vow_idx = words[k]['vowel_idxs'][j]
            vow_cats[vow_idx] = new_cat

def resample_lexical_items(words, ncons, anneal, vowels, phonemes, 
                           lexical_items, vow_cats, lex_cats):
    # Loop through all lexical items
    for i, lex_item in enumerate(lexical_items):
        # Look through vowels in lexical item
        for j, phon_idx in enumerate(lex_item['phoneme_idxs']):
            phoneme_mean, phoneme_ss = remove_vowel(
                j, phon_idx, i, lex_item, lexical_items, words, phonemes, 
                vow_cats, lex_cats
            )
            add_vowel(
                j, phoneme_mean, phoneme_ss, i, lex_item, vow_cats, lex_cats,
                phonemes, anneal, params
            )

def gibbs_sample(words, vowels, params, num_samples=10000, print_every=10000):
    """
    Kicks off the Gibbs sampling process
    """
    ##################
    # INITIALIZATION #
    ##################
    print("Initializing parameters")
    lex_cats = torch.zeros([len(words)]) - 1
    vow_cats = torch.zeros([vowels.shape[0]]) - 1
    anneal = get_annealing_factor(0, params)

    phonemes = {
        'cat_mus': [],
        'cat_covs': [],
        'cat_nus': [],
        'cat_counts': [],
        'max_cat': -1
    }

    params['dims_tensor'] = torch.arange(vowels.shape[1])[:, None]

    lexical_items = []

    ncons = defaultdict(int)

    # Initialize lexical items/phonemes
    for w, word in enumerate(words):
        # Sample lexical items
        add_word(
            w, word, ncons, anneal, vowels, phonemes, lexical_items, vow_cats, 
            lex_cats, params
        )

    print(lexical_items)
    print(phonemes)
    
    # Start sampling
    for b in range(num_samples):
        print("Iteration {}".format(b))
        anneal = get_annealing_factor(b, params)

        print("Resampling lexical items")
        resample_lexical_items(
            words, ncons, anneal, vowels, phonemes, lexical_items, vow_cats, 
            lex_cats
        )
        print("Resampling words")
        resample_words(
            words, ncons, anneal, vowels, phonemes, lexical_items, vow_cats, 
            lex_cats
        )

        if b % print_every == 0:
            #ll = get_joint_probability(x, z, phonemes, params)
            #print(ll)
            pprint(phonemes)

    return phonemes, lexical_items, vow_cats, lex_cats

if __name__ == "__main__":

    # A dictionary to hold the simulation parameters
    params = {
        #########################################
        # PARAMS RELATED TO SAMPLING INPUT DATA #
        #########################################
        # Number of word_samples
        'N': 5,

        # Files for sampling input data
        'vowel_file': 'corpus_data/hillenbrand_vowel_acoustics.csv',
        'word_file': 'corpus_data/childes_counts_prons.csv',

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

        ########################################
        # PARAMETERS RELATED TO GIBBS SAMPLING #
        ########################################

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
        # Alpha for dirichlet process on phonemes
        'alpha': 10,

        # Beta for dirichlet process on words
        'beta': 10000,

        # Annealing factor
        'annealing_factor': 9,

        # Number of lexical items to sample for likelihood approximation
        'num_lexical_samples': 100,

        # geometric constant controlling word length
        'geometric': 1/2,

        # Probability of selecting a consonant (instead of vowel)
        'cons': 0.62,

        # Default parameters for category distributions
        #'mu_0': torch.tensor([500., 1500.]),
        'S_0': torch.eye(2),
        'nu_0': 1.001
    }

    # Set mu_0 to contain only dimensions we're using
    params['mu_0'] = torch.Tensor([
        params['prior_means'][dim] for dim in params['dimensions']
    ])

    for i in range(289, 1000000):
        SEED = i
        random.seed(SEED)
        torch.manual_seed(SEED)

        words, vowel_samples, vowel_labels = sample_inputs(
            params['vowel_file'], params['word_file'], params['dimensions'],
            num_samples=params['N']
        )

        # words = [
        #     {'word': 'foo', 
        #      'pron': 'f oo', 
        #      'template': 'f _', 
        #      'vowel_idxs': [0], 
        #      'vowels': ['oo'], 
        #      'cons': ['f'],
        #      'len': 2,
        #      'vowel_len': 1,
        #      'cons_len': 1
        #     },
        #     {'word': 'baa', 
        #      'pron': 'b aa', 
        #      'template': 'b _', 
        #      'vowel_idxs': [1], 
        #      'vowels': ['aa'], 
        #      'cons': ['b'],
        #      'len': 2,
        #      'vowel_len': 1,
        #      'cons_len': 1
        #     }
        # ]
        # words = [
        #     {'word': 'booboo', 
        #      'pron': 'b oo b oo', 
        #      'template': 'b _ b _', 
        #      'vowel_idxs': [0, 1], 
        #      'vowels': ['oo', 'oo'], 
        #      'cons': ['b', 'b'],
        #      'len': 4,
        #      'vowel_len': 2,
        #      'cons_len': 2
        #     },
        # ]

        # vowel_samples = [
        #     torch.Tensor([501, 3001]),
        #     torch.Tensor([450, 2800])
        # ]

        print(words)
        print(vowel_samples)

        # try:
        phonemes, lexical_items, z, l = gibbs_sample(
            words, torch.stack(vowel_samples), params, print_every=100, 
            num_samples=5
        )
        # except:
        #     import pdb; pdb.set_trace()

        # seed = 17, i 6, n=5, lex=100
        # seed = 65, i = 3, n=5, lex=100

        # seed 289, i = 0, N=5, lex=100