library(ellipse)
library(rags2ridges)
library(tidyverse)

plot_distributions <- function(mu, cov) {
  ggplot(data=mu, aes(x=f2, y=f1, label=vowel)) +
    scale_x_reverse() +
    scale_y_reverse() +
    geom_text()
  
  graph <- ggplot()
  
  cov <- cov %>% 
    select('vowel', 'f1-f1', 'f1-f2', 'f2-f1', 'f2-f2')
  
  el = c()
  
  for (i in 1:nrow(mu)) {
    cov_vals <- matrix(as.numeric(cov[i, 2:ncol(cov)]), ncol=2)
    v_el <- ellipse(cov_vals, centre=matrix(mu[i,] %>% select('f1', 'f2') %>% as.numeric))
    v_el <- as_tibble(v_el)
    v_el$vowel <- mu[i,]$vowel
    el <- rbind(el, v_el)
  }
  graph <- graph + geom_point(data=el, aes(x=y, y=x, color=vowel), size=4)
  graph + 
    geom_label(data=mu, aes(x=f2, y=f1, label=vowel, color=vowel), size=10) +
    scale_x_reverse() +
    scale_y_reverse() +
    xlab("F2 (Barks)") +
    ylab("F1 (Barks)") +
    theme(legend.position="none", text=element_text(size=30))
}

get_kls <- function(mus, covs, my_dims, name) {
  name <- str_c(c(name, my_dims), collapse="_")
  mus <- mus %>% 
    select(c(my_dims, 'vowel'))
  
  cov_cols <- c()
  for (i in my_dims) {
    for (j in my_dims) {
      cov_cols <- c(cov_cols, str_c(i, j, sep='-'))
    }
  }
  covs <- covs %>%
    select(c(all_of(cov_cols), 'vowel'))
  
  ndims <- length(my_dims)
  
  kl_mat <- matrix(0, nrow(mus), nrow(mus))
  colnames(kl_mat) <- mus$vowel
  rownames(kl_mat) <- mus$vowel
  
  for (i in (1:nrow(mus))) {
    for (j in (1:nrow(mus))) {
      mu1 <- unlist(as.data.frame(mus[i, 1:ndims]))
      mu2 <- unlist(as.data.frame(mus[j, 1:ndims]))
      names(mu1) <- my_dims
      names(mu2) <- my_dims
      cov1 <- data.matrix(as.data.frame(matrix(covs[i, 1:(ndims^2)], ncol=ndims)))
      cov2 <- data.matrix(as.data.frame(matrix(covs[j, 1:(ndims^2)], ncol=ndims)))
      colnames(cov1) <- my_dims
      rownames(cov1) <- my_dims
      colnames(cov2) <- my_dims
      rownames(cov2) <- my_dims
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
  full_kls <- kl_mat
  
  full_kls <- full_kls %>%
    as.data.frame() %>%
    rownames_to_column("true_vowel") %>%
    pivot_longer(-c(true_vowel), names_to='app_vowel', values_to='kl') %>%
    mutate(kl = round(kl, 2))
  
  full_kls %>%
    ggplot(aes(y=true_vowel, x=app_vowel, fill=kl)) +
    geom_tile() +
    scale_fill_distiller() +
    geom_text(aes(label=kl), size=5) +
    xlab("Q") + 
    ylab("P") +
    theme(axis.text=element_text(size=30),
          axis.title=element_text(size=30),
          legend.title=element_text(size=10),
          legend.text=element_text(size=10)) +
    guides(fill=guide_legend(title="Dkl(P||Q)"))
  ggsave(file.path(FIGURES_DIR, str_c(name, '_kl_full.png')))
  
  sym_kls <- sym_kls %>%
    as.data.frame() %>%
    rownames_to_column("v1") %>%
    pivot_longer(-c(v1), names_to='v2', values_to='kl') %>%
    mutate(kl = round(kl, 2))
  
  sym_kls %>%
    ggplot(aes(y=v1, x=v2, fill=kl)) +
    geom_tile() +
    scale_fill_distiller() +
    geom_text(aes(label=kl), size=5) +
    xlab("Vowel 1") +
    ylab("Vowel 2") + 
    theme(axis.text=element_text(size=30),
          axis.title=element_text(size=30),
          legend.title=element_text(size=10),
          legend.text=element_text(size=10),
          legend.position="none") +
    guides(fill=guide_legend(title="Dkl(P||Q) + Dkl(Q||P)"))
  ggsave(file.path(FIGURES_DIR, str_c(name, '_kl_symmetrical.png')))
}

compare_kls_register <- function(train_mus, train_covs, test_mus, test_covs, my_dims) {
  train_mus <- train_mus %>% 
    select(c(my_dims, 'vowel'))
  
  test_mus <- test_mus %>% 
    select(c(my_dims, 'vowel'))
  
  cov_cols <- c()
  for (i in my_dims) {
    for (j in my_dims) {
      cov_cols <- c(cov_cols, str_c(i, j, sep='-'))
    }
  }
  train_covs <- train_covs %>%
    select(c(all_of(cov_cols), 'vowel'))
  
  test_covs <- test_covs %>%
    select(c(all_of(cov_cols), 'vowel'))
  
  ndims <- length(my_dims)
  
  kl_mat <- matrix(0, 1, nrow(train_mus))
  colnames(kl_mat) <- train_mus$vowel
  
  for (i in (1:nrow(train_mus))) {
    mu1 <- unlist(as.data.frame(train_mus[i, 1:ndims]))
    mu2 <- unlist(as.data.frame(test_mus[i, 1:ndims]))
    names(mu1) <- my_dims
    names(mu2) <- my_dims
    cov1 <- data.matrix(as.data.frame(matrix(train_covs[i, 1:(ndims^2)], ncol=ndims)))
    cov2 <- data.matrix(as.data.frame(matrix(test_covs[i, 1:(ndims^2)], ncol=ndims)))
    colnames(cov1) <- my_dims
    rownames(cov1) <- my_dims
    colnames(cov2) <- my_dims
    rownames(cov2) <- my_dims
    # mu1 is 'approximate', mu2 is 'true'
    kl_mat[i] <- KLdiv(mu1, mu2, cov1, cov2)
  }
  return(kl_mat)
}

get_ads_cds_comparison <- function(train_mu_1, train_cov_1, train_mu_2, 
                                   train_cov_2, test_mu, test_cov, my_dims, 
                                   name) {
  kl_ads_ads <- compare_kls_register(
    train_mu_1, train_cov_1, test_mu, test_cov, my_dims 
  )
  kl_cds_ads <- compare_kls_register(
    train_mu_2, train_cov_2, test_mu, test_cov, my_dims
  )
  
  kl_ads_ads <- kl_ads_ads %>% 
    as_tibble() %>%
    pivot_longer(everything(), names_to='vowel', values_to="kl") %>%
    mutate(group='ads-ads')
  
  kl_cds_ads <- kl_cds_ads %>% 
    as_tibble() %>%
    pivot_longer(everything(), names_to='vowel', values_to="kl") %>%
    mutate(group='cds-ads')
  
  kl_comparison <- rbind(kl_ads_ads, kl_cds_ads)
  
  ggplot(kl_comparison) +
    geom_bar(aes(x=vowel, y=kl, fill=group), stat='identity', position='dodge') +
    xlab("Vowel") +
    ylab("KL Divergence") +
    theme(axis.text=element_text(size=20),
          axis.title=element_text(size=20),
          legend.title=element_text(size=10),
          legend.text=element_text(size=10)) +
    guides(fill=guide_legend(title="Comparison group"))
  ggsave(file.path(FIGURES_DIR, str_c(name, '.png')))
}

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/training_data/distributions')

my_dims <- c('f1', 'f2', 'f3', 'duration')
en_train_ads_mu <- read_csv('english_train_ads_mu.csv')
en_train_ads_cov <- read_csv('english_train_ads_cov.csv')
#plot_distributions(en_train_ads_mu, en_train_ads_cov)
ggsave('../../figures/poster/en_train_ads_distributions.png')
get_kls(en_train_ads_mu, en_train_ads_cov, my_dims, name='en_train_ads')

en_test_ads_mu <- read_csv('english_test_ads_mu.csv')
en_test_ads_cov <- read_csv('english_test_ads_cov.csv')
plot_distributions(en_test_ads_mu, en_test_ads_cov)
ggsave('../../figures/poster/en_test_ads_distributions.png')
get_kls(en_test_ads_mu, en_test_ads_cov, my_dims, name='en_test_ads')

en_train_cds_mu <- read_csv('english_train_cds_mu.csv')
en_train_cds_cov <- read_csv('english_train_cds_cov.csv')
plot_distributions(en_train_cds_mu, en_train_cds_cov)
ggsave('../../figures/poster/en_train_cds_distributions.png')
get_kls(en_train_cds_mu, en_train_cds_cov, my_dims, name='en_train_cds')

sp_train_ads_mu <- read_csv('spanish_train_ads_mu.csv')
sp_train_ads_cov <- read_csv('spanish_train_ads_cov.csv')
plot_distributions(sp_train_ads_mu, sp_train_ads_cov)
ggsave('../../figures/poster/sp_train_ads_distributions.png')
get_kls(sp_train_ads_mu, sp_train_ads_cov, my_dims, name='sp_train_ads')

sp_test_ads_mu <- read_csv('spanish_test_ads_mu.csv')
sp_test_ads_cov <- read_csv('spanish_test_ads_cov.csv')
plot_distributions(sp_test_ads_mu, sp_test_ads_cov)
ggsave('../../figures/poster/sp_test_ads_distributions.png')
get_kls(sp_test_ads_mu, sp_test_ads_cov, my_dims, name='sp_test_ads')

sp_train_cds_mu <- read_csv('spanish_train_cds_mu.csv')
sp_train_cds_cov <- read_csv('spanish_train_cds_cov.csv')
plot_distributions(sp_train_cds_mu, sp_train_cds_cov)
ggsave('../../figures/poster/sp_train_cds_distributions.png')
get_kls(sp_train_cds_mu, sp_train_cds_cov, my_dims, name='sp_train_cds')

get_ads_cds_comparison(
  sp_train_ads_mu, sp_train_ads_cov, sp_train_cds_mu, sp_train_cds_cov,
  sp_test_ads_mu, sp_test_ads_cov, my_dims, name="kl_comparison_spanish"
)

get_ads_cds_comparison(
  sp_train_ads_mu, sp_train_ads_cov, sp_train_cds_mu, sp_train_cds_cov,
  sp_train_cds_mu, sp_train_cds_cov, my_dims, name="kl_comparison_spanish_cds"
)

get_ads_cds_comparison(
  en_train_ads_mu, en_train_ads_cov, en_train_cds_mu, en_train_cds_cov,
  en_test_ads_mu, en_test_ads_cov, my_dims, name="kl_comparison_english"
)

get_ads_cds_comparison(
  en_train_ads_mu, en_train_ads_cov, en_train_cds_mu, en_train_cds_cov,
  en_train_cds_mu, en_train_cds_cov, my_dims, name="kl_comparison_english_cds"
)

# # Compare ADS vs CDS English
kl_ads_cds <- compare_kls_register(
  en_train_ads_mu, en_train_ads_cov, en_train_cds_mu, en_train_cds_cov,
  my_dims
)
kl_cds_ads <- compare_kls_register(
  en_train_cds_mu, en_train_cds_cov, en_train_ads_mu, en_train_ads_cov,
  my_dims
)

kl_ads_cds <- kl_ads_cds %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to='vowel', values_to="kl") %>%
  mutate(group='ads-cds')

kl_cds_ads <- kl_cds_ads %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to='vowel', values_to="kl") %>%
  mutate(group='cds-ads')

kl_comparison <- rbind(kl_ads_cds, kl_cds_ads)

ggplot(kl_comparison) +
  geom_bar(aes(x=vowel, y=kl, fill=group), stat='identity', position='dodge') +
  xlab("Vowel") +
  ylab("KL Divergence") +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=20),
        legend.title=element_text(size=10),
        legend.text=element_text(size=10)) +
  guides(fill=guide_legend(title="Comparison group"))
ggsave(file.path(FIGURES_DIR, 'kl_comparison_bidirectional_english.png'))


# # Compare ADS vs CDS English
kl_ads_cds <- compare_kls_register(
  sp_train_ads_mu, sp_train_ads_cov, sp_train_cds_mu, sp_train_cds_cov,
  my_dims
)
kl_cds_ads <- compare_kls_register(
  sp_train_cds_mu, sp_train_cds_cov, sp_train_ads_mu, sp_train_ads_cov,
  my_dims
)

kl_ads_cds <- kl_ads_cds %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to='vowel', values_to="kl") %>%
  mutate(group='ads-cds')

kl_cds_ads <- kl_cds_ads %>%
  as_tibble() %>%
  pivot_longer(everything(), names_to='vowel', values_to="kl") %>%
  mutate(group='cds-ads')

kl_comparison <- rbind(kl_ads_cds, kl_cds_ads)

ggplot(kl_comparison) +
  geom_bar(aes(x=vowel, y=kl, fill=group), stat='identity', position='dodge') +
  xlab("Vowel") +
  ylab("KL Divergence") +
  theme(axis.text=element_text(size=20),
        axis.title=element_text(size=20),
        legend.title=element_text(size=10),
        legend.text=element_text(size=10)) +
  guides(fill=guide_legend(title="Comparison group"))
ggsave(file.path(FIGURES_DIR, 'kl_comparison_bidirectional_spanish.png'))