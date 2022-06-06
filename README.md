# Nonparametric Bayesian vowel learners

This repository contains Python implementations of the vowel learners in Feldman et al. (2013).

The `src` directory contains the code:
* `distributional_learner.py` and `interactive_learner.py` contain implementations of the distributional and interactive learners. The point of entry to these learners is the `run` method, which expects a list of vowel samples (in some number of dimensions) and a dictionary of parameters.
* `run_simulation.py` runs a simulation. This script samples input from a set of vowel distributions according to their frequency and then runs the distributional learner on the sampled tokens (I'll modify this later to also include the interactive learner). This script five positional arguments:
  * `mu_file`: the filename of the file containing category means
  * `cov_file`: the filename of the file containing category covariance matrices
  * `counts_file`: the filename of the file containing category covariance matrices
  * `intput_folder`: the folder containing the input files (the above three)
  * `output_folder`: the folder where the output will be saved. This folder must exist.

  There are a number of optional arguments you can view by running `python run_simulation.py --help`.
  
  This script will output four files:
    * A CSV file contaning the acoustics of the sampled vowels, their true categories, and their learned categories
    * A CSV containing the means of the learned categories
    * A CSV containing the covariance matrices of the learned categories
    * A CSV containing the log likelihood of the model by iteration

The `src/data_processing` directory contains R scripts for visualizing or analyzing the data. These are disorganized right now, but you can take a look at them.

The `training_data` directory has the various input data we've been using. The most relevant ehre is the `distributions` folder, which contains the English/Spanish ADS/CDS distribution files in both Barks and Hz.

## Examples of running the learner

An example of how to run the learned on Spanish ADS speech in Hz on F1 and F2:

```python run_simulation spanish_train_ads_mu.csv spanish_train_ads_cov.csv spanish_train_ads_counts.csv ../training_data/distributions/hz ../model_outputs```

Running the same model using Barks

```python run_simulation spanish_train_ads_mu.csv spanish_train_ads_cov.csv spanish_train_ads_counts.csv ../training_data/distributions ../model_outputs --barks```

Running the same model using Barks

```python run_simulation spanish_train_ads_mu.csv spanish_train_ads_cov.csv spanish_train_ads_counts.csv ../training_data/distributions ../model_outputs --barks```

Running the model using more dimensions 

```python run_simulation spanish_train_ads_mu.csv spanish_train_ads_cov.csv spanish_train_ads_counts.csv ../training_data/distributions ../model_outputs --barks --dims f1 f2 f3 duration```

Running the model using more vowel samples

```python run_simulation spanish_train_ads_mu.csv spanish_train_ads_cov.csv spanish_train_ads_counts.csv ../training_data/distributions ../model_outputs --barks --vowel_samples 10000```
