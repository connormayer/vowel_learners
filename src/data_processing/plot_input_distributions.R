library(ellipse)
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
  graph <- graph + geom_point(data=el, aes(x=y, y=x, color=vowel), size=3)
  graph + 
    geom_label(data=mu, aes(x=f2, y=f1, label=vowel, color=vowel)) +
    scale_x_reverse() +
    scale_y_reverse() +
    xlab("F2") +
    ylab("F1") +
    theme(legend.position="none")
}

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/training_data/distributions')

en_train_ads_mu <- read_csv('english_train_ads_mu.csv')
en_train_ads_cov <- read_csv('english_train_ads_cov.csv')
plot_distributions(en_train_ads_mu, en_train_ads_cov)
ggsave('../../figures/en_train_ads_distributions.png')

en_test_ads_mu <- read_csv('english_test_ads_mu.csv')
en_test_ads_cov <- read_csv('english_test_ads_cov.csv')
plot_distributions(en_test_ads_mu, en_test_ads_cov)
ggsave('../../figures/en_test_ads_distributions.png')

en_train_cds_mu <- read_csv('english_train_cds_mu.csv')
en_train_cds_cov <- read_csv('english_train_cds_cov.csv')
plot_distributions(en_train_cds_mu, en_train_cds_cov)
ggsave('../../figures/en_train_cds_distributions.png')

sp_train_ads_mu <- read_csv('spanish_train_ads_mu.csv')
sp_train_ads_cov <- read_csv('spanish_train_ads_cov.csv')
plot_distributions(sp_train_ads_mu, sp_train_ads_cov)
ggsave('../../figures/sp_train_ads_distributions.png')

sp_test_ads_mu <- read_csv('spanish_test_ads_mu.csv')
sp_test_ads_cov <- read_csv('spanish_test_ads_cov.csv')
plot_distributions(sp_test_ads_mu, sp_test_ads_cov)
ggsave('../../figures/sp_test_ads_distributions.png')

sp_train_cds_mu <- read_csv('spanish_train_cds_mu.csv')
sp_train_cds_cov <- read_csv('spanish_train_cds_cov.csv')
plot_distributions(sp_train_cds_mu, sp_train_cds_cov)
ggsave('../../figures/sp_train_cds_distributions.png')


