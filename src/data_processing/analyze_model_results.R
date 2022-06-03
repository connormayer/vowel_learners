library(ellipse)
library(rags2ridges)
library(tidyverse)

FIGURES_DIR <- 'C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/figures/poster'

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
    geom_label(data=mu, aes(x=f2, y=f1, label=vowel, color=as.factor(vowel)), size=12) +
    scale_x_reverse() +
    scale_y_reverse() +
    xlab("F2 (Barks)") +
    ylab("F1 (Barks)") +
    theme(legend.position="none", text=element_text(size=30))
}

get_kls <- function(mus, covs, results) {
  dims <- names(mus)[1:length(mus) - 1]
  ndims <- length(dims)
  
  kl_mat <- matrix(0, nrow(mus), nrow(mus))
  colnames(kl_mat) <- mus$vowel
  rownames(kl_mat) <- mus$vowel
  
  for (i in (mus$vowel + 1)) {
    for (j in (mus$vowel + 1)) {
      mu1 <- unlist(as.data.frame(mus[i, 1:ndims]))
      mu2 <- unlist(as.data.frame(mus[j, 1:ndims]))
      names(mu1) <- dims
      names(mu2) <- dims
      cov1 <- data.matrix(as.data.frame(matrix(covs[i, 1:(ndims^2)], ncol=ndims)))
      cov2 <- data.matrix(as.data.frame(matrix(covs[j, 1:(ndims^2)], ncol=ndims)))
      colnames(cov1) <- dims
      rownames(cov1) <- dims
      colnames(cov2) <- dims
      rownames(cov2) <- dims
      # mu1 is 'true', mu2 is 'approximate'
      kl_mat[i,j] <- KLdiv(mu2, mu1, cov2, cov1)
    }
  }
  
  sym_kls <- kl_mat
  for (i in 1:(nrow(kl_mat) - 1)) {
    for (j in (i+1):ncol(sym_kls)) {
      sym_kls[i,j] <- kl_mat[i,j] + kl_mat[j,i]
      sym_kls[j,i] <- NA
    }
  }
  return(list(kl_mat, sym_kls))
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
  ggsave(file.path(FIGURES_DIR, str_c(name, '_true_cats.png')), dpi=300)
  
  ggplot(results, aes(x=f2, y=f1, color=as.factor(learned_cat))) +
    geom_point(size=3, alpha=0.5) +
    scale_x_reverse() +
    scale_y_reverse() +
    labs(color="Learned category")
  ggsave(file.path(FIGURES_DIR, str_c(name, '_learned_cats.png')), dpi=300)
  
  # Look at learned distributions
  mus <- read_csv(mu_file)
  covs <- read_csv(cov_file)
  plot_distributions(mus, covs, results)
  ggsave(file.path(FIGURES_DIR, str_c(name, '_learned_cat_distributions.png')), dpi=300)
  
  # Look at LL
  ll <- read_csv(ll_file)
  ggplot(ll, aes(x=iteration, y=log_likelihood)) +
    geom_line()
  ggsave(file.path(FIGURES_DIR, str_c(name, '_ll.png')), dpi=300)
  
  # Calculate KL divergence
  kls = get_kls(mus, covs, results)
  full_kls <- kls[[1]]
  sym_kls <- kls[[2]]
  
  full_kls <- full_kls %>%
    as.data.frame() %>%
    rownames_to_column("true_vowel") %>%
    pivot_longer(-c(true_vowel), names_to='app_vowel', values_to='kl') %>%
    mutate(kl = round(kl, 2))
  
  full_kls %>%
    ggplot(aes(y=true_vowel, x=app_vowel, fill=kl)) +
    geom_tile() +
    scale_fill_distiller() +
    geom_text(aes(label=kl), size=10) +
    xlab("Q") + 
    ylab("P") +
    theme(axis.text=element_text(size=20),
          axis.title=element_text(size=20)) +
    guides(fill=guide_legend(title="Dkl(P||Q)"))
  ggsave(file.path(FIGURES_DIR, str_c(name, '_learned_kl_full.png')), dpi=300)
  
  sym_kls <- sym_kls %>%
    as.data.frame() %>%
    rownames_to_column("v1") %>%
    pivot_longer(-c(v1), names_to='v2', values_to='kl') %>%
    mutate(kl = round(kl, 2))
  
  sym_kls %>%
    ggplot(aes(y=v1, x=v2, fill=kl)) +
    geom_tile() +
    scale_fill_distiller() +
    geom_text(aes(label=kl), size=10) +
    xlab("Vowel 1") +
    ylab("Vowel 2") + 
    theme(axis.text=element_text(size=20),
          axis.title=element_text(size=20),
          legend.title=element_text(size=20),
          legend.text=element_text(size=15)) +
    guides(fill=guide_legend(title="Dkl(P||Q) + Dkl(Q||P)"))
  ggsave(file.path(FIGURES_DIR, str_c(name, '_learned_kl_symmetrical.png')), dpi=300)
}

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/model_outputs/high_sample_barks')

ads_file <- 'english_ads_f1_f2_f3_duration.csv'
ads_mu_file <- 'english_ads_mus_f1_f2_f3_duration.csv'
ads_cov_file <- 'english_ads_covs_f1_f2_f3_duration.csv'
ads_ll_file <- 'english_ads_ll_f1_f2_f3_duration.csv'

analyze_results(ads_file, ads_mu_file, ads_cov_file, ads_ll_file)

cds_file <- 'english_cds_f1_f2_f3_duration.csv'
cds_mu_file <- 'english_cds_mus_f1_f2_f3_duration.csv'
cds_cov_file <- 'english_cds_covs_f1_f2_f3_duration.csv'
cds_ll_file <- 'english_cds_ll_f1_f2_f3_duration.csv'

analyze_results(cds_file, cds_mu_file, cds_cov_file, cds_ll_file)


