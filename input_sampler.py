import pandas as pd
import torch

# Represent diphthongs as pairs of vowels
# Clean up some weird representations
VOWEL_MAPPINGS = {
    'aa': 'ah',
    'ao': 'aw',
    'ey': 'ei',
    'ow': 'oa',
    'uh': 'oo',
    'ah': 'uh',
    'ay': 'ah ih',
    'oy': 'aw ih',
    'aw': 'ah oo'
}

def create_vowel_distributions(vowel_file, dimensions):
    # This assumes vowel file is formatted like
    # <VOWEL>, <mean|std>, <dim1>, <dim2>, ...
    # The second column contains either 'mean' or 'std' and indicates
    # whether the values for each dim in the row are the mean or std
    # This means each vowel corresponds to two rows. There can be
    # any number of dimensions
    vowel_acoustics = pd.read_csv(vowel_file)
    dist_dict = {}
    for v in vowel_acoustics.vowel.unique():
        v_rows = vowel_acoustics[vowel_acoustics['vowel'] == v][['measurement'] + dimensions]
        dist_dict[v] = torch.distributions.Normal(
            torch.Tensor(v_rows[v_rows['measurement'] == 'mean'].iloc[:, 1:].values),
            torch.Tensor(v_rows[v_rows['measurement'] == 'std'].iloc[:, 1:].values)
        )
    return dist_dict

def process_prons(pron):
    pron = pron.lower()
    pron = pron.split(' ')
    pron = [VOWEL_MAPPINGS.get(s, s) for s in pron]
    return ' '.join(pron)

def sample_words(word_file, vowel_dists, num_words=0, num_samples=5000, 
                 ensure_minimum=False, replace=True):
    """
    word_file: file containing words, counts, and pronunciation columns
    vowel_dists: A dictionary of acoustic vowel distributions
    num_words: The number of unique word types to sample from. The n most frequent words
               are chosen. 
    num_samples: Total number of samples to take
    ensure_minimum: Forces sampling of at least one word per vowel
    replace: Sample words with replacement>
    """
    words = pd.read_csv(word_file)
    words = words.sort_values('count', ascending=False)
    words['pronunciation'] = words['pronunciation'].apply(process_prons)

    if ensure_minimum:
        # worry about this later
        pass

    max_words = num_words if num_words > 0 else words.shape[0]
    words = words[:max_words]
    sampled_word_idxs = torch.multinomial(
        torch.Tensor(words['count'].values),
        num_samples,
        replacement=replace

    )

    sampled_words = []
    sampled_vowels = []
    vowel_labels = []
    vowel_idx = 0

    for w_idx in sampled_word_idxs:
        word_dict = {}
        word = words.iloc[w_idx.item()]

        # Orthographic representation
        word_dict['word'] = word['word']

        # Pronunciation
        word_dict['pron'] = word['pronunciation']

        # Template (vowels removed)
        split_pron = word['pronunciation'].split(' ')
        template = list(
            map(lambda x: '_' if x in vowel_dists.keys() else x, split_pron)
        )
        word_dict['cons'] = list(filter(lambda x: x != '_', template))
        word_dict['cons_len'] = len(word_dict['cons'])
        word_dict['template'] = ' '.join(template)

        # Sample vowels
        vowels = [s for s in split_pron if s in vowel_dists.keys()]
        vowel_idxs = []
        for v in vowels:
            sampled_vowels.append(vowel_dists[v].sample()[0])
            vowel_labels.append(v)
            vowel_idxs.append(vowel_idx)
            vowel_idx += 1
        word_dict['vowel_idxs'] = vowel_idxs
        word_dict['vowels'] = vowels
        word_dict['vowel_len'] = len(vowels)
        word_dict['len'] = len(split_pron)

        sampled_words.append(word_dict)

    return sampled_words, sampled_vowels, vowel_labels

def sample_inputs(vowel_file, word_file, dimensions, num_words=0, num_samples=5000,
                  ensure_minimum=False, replace=True):
    vowel_dists = create_vowel_distributions(vowel_file, dimensions)
    words, vowel_samples, vowel_labels = sample_words(
        word_file, vowel_dists, num_words, num_samples, ensure_minimum, replace
    )
    return words, vowel_samples, vowel_labels    

if __name__ == "__main__":
    vowel_file = 'corpus_data/hillenbrand_vowel_acoustics.csv'
    word_file = 'corpus_data/childes_counts_prons.csv'
    words, vowel_samples, vowel_labels = sample_inputs(vowel_file, word_file)
