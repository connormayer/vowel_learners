library(ellipse)
library(tidyverse)

FIGURES_DIR = 'C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/figures/'

plot_distributions <- function(mu, cov, results) {
  ggplot(data=mu, aes(x=f2, y=f1, label=vowel)) +
    scale_x_reverse() +
    scale_y_reverse() +
    geom_text()
  
  graph <- ggplot()
  
  cov <- cov %>% 
    select('vowel', 'f1-f1', 'f1-f2', 'f2-f1', 'f2-f2')
  
  el = c()
  
  for (i in 1:nrow(mu)) {
    cov_vals <- matrix(as.numeric(cov[i, 2:ncol(cov)]), ncol=2) / nrow(results %>% filter(learned_cat == i - 1))
    v_el <- ellipse(cov_vals, centre=matrix(mu[i,] %>% select('f1', 'f2') %>% as.numeric))
    v_el <- as_tibble(v_el)
    v_el$vowel <- mu[i,]$vowel
    el <- rbind(el, v_el)
  }
  graph <- graph + geom_point(data=el, aes(x=y, y=x, color=as.factor(vowel)), size=4)
  graph + 
    geom_label(data=mu, aes(x=f2, y=f1, label=vowel, color=as.factor(vowel)), size=4) +
    scale_x_reverse() +
    scale_y_reverse() +
    xlab("F2") +
    ylab("F1") +
    theme(legend.position="none")
}

analyze_results <- function(results_file, mu_file, cov_file, ll_file) {
  
  name = strsplit(results_file, "\\.")[[1]][1]
  
  # Look at true vs. learned category assignments
  results <- read_csv(results_file)
  ggplot(results, aes(x=f2, y=f1, color=as.factor(vowel))) +
    geom_point(size=3) +
    scale_x_reverse() +
    scale_y_reverse() +
    labs(color="Vowel")
  ggsave(file.path(FIGURES_DIR, str_c(name, '_true_cats.png')))
  
  ggplot(results, aes(x=f2, y=f1, color=as.factor(learned_cat))) +
    geom_point(size=3, alpha=0.5) +
    scale_x_reverse() +
    scale_y_reverse() +
    labs(color="Learned category")
  ggsave(file.path(FIGURES_DIR, str_c(name, '_learned_cats.png')))
  
  # Look at learned distributions
  mus <- read_csv(mu_file)
  covs <- read_csv(cov_file)
  plot_distributions(mus, covs, results)
  ggsave(file.path(FIGURES_DIR, str_c(name, '_learned_cat_distributions.png')))
  
  # Look at LL
  ll <- read_csv(ll_file)
  ggplot(ll, aes(x=iteration, y=log_likelihood)) +
    geom_line()
  ggsave(file.path(FIGURES_DIR, str_c(name, 'll.png')))
  
  
}

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/model_outputs')

# Look at Spanish ADS
ads_file <- 'spanish_ads_f1_f2.csv'
ads_mu_file <- 'spanish_ads_mus_f1_f2.csv'
ads_cov_file <- 'spanish_ads_covs_f1_f2.csv'
ads_ll_file <- 'spanish_ads_ll_f1_f2.csv'

analyze_results(ads_file, ads_mu_file, ads_cov_file, ads_ll_file)

# Look at Spanish CDS
cds_file <- 'spanish_cds_f1_f2.csv'
cds_mu_file <- 'spanish_cds_mus_f1_f2.csv'
cds_cov_file <- 'spanish_cds_covs_f1_f2.csv'
cds_ll_file <- 'spanish_cds_ll_f1_f2.csv'

analyze_results(cds_file, cds_mu_file, cds_cov_file, cds_ll_file)

# Look at English ADS
ads_file <- 'english_ads_f1_f2.csv'
ads_mu_file <- 'english_ads_mus_f1_f2.csv'
ads_cov_file <- 'english_ads_covs_f1_f2.csv'
ads_ll_file <- 'english_ads_ll_f1_f2.csv'

analyze_results(ads_file, ads_mu_file, ads_cov_file, ads_ll_file)

# Look at English CDS
cds_file <- 'english_cds_f1_f2.csv'
cds_mu_file <- 'english_cds_mus_f1_f2.csv'
cds_cov_file <- 'english_cds_covs_f1_f2.csv'
cds_ll_file <- 'english_cds_ll_f1_f2.csv'

analyze_results(cds_file, cds_mu_file, cds_cov_file, cds_ll_file)


