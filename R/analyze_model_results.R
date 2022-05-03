library(ellipse)
library(tidyverse)

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/outputs/spanish_ads_cds/')
ads_filename <- 'ads_results_f1_f2.csv'
ads <- read_csv(ads_filename)

ggplot(ads, aes(x=f2, y=f1, color=as.factor(learned_cats))) +
  geom_point(size=3) +
  scale_x_reverse() +
  scale_y_reverse()

ads_mu_file <- 'ads_mus.csv'
ads_mus <- read_csv(ads_mu_file, col_names=FALSE)

ads_cov_file <- 'ads_covs.csv'
ads_covs <- read_csv(ads_cov_file, col_names=FALSE)

ggplot(data=ads_mus, aes(x=X2, y=X1)) +
  geom_point(size=3) +
  scale_x_reverse() +
  scale_y_reverse()

graph <- ggplot()

for (i in 1:nrow(ads_covs)) {
  cov <- matrix(as.numeric(ads_covs[i,]), ncol=2) / nrow(ads %>% filter(learned_cats == i - 1))
  el <- ellipse(cov, centre=matrix(as.numeric(ads_mus[i,])))
  graph <- graph + geom_point(data=as_tibble(el), aes(x=y, y=x))
  graph
}
graph + 
  scale_x_reverse() +
  scale_y_reverse()



cds_filename <- 'cds_results_f1_f2.csv'

cds <- read_csv(cds_filename)

ggplot(cds, aes(x=f2, y=f1, color=as.factor(learned_cats))) +
  geom_point(size=3) +
  scale_x_reverse() +
  scale_y_reverse()

cds_mu_file <- 'cds_mus.csv'
cds_mus <- read_csv(cds_mu_file, col_names=FALSE)

cds_cov_file <- 'cds_covs.csv'
cds_covs <- read_csv(cds_cov_file, col_names=FALSE)

ggplot(data=cds_mus, aes(x=X2, y=X1)) +
  geom_point(size=3) +
  scale_x_reverse() +
  scale_y_reverse()

graph <- ggplot()

for (i in 1:length(cds_covs)) {
  cov <- matrix(as.numeric(cds_covs[i,]), ncol=2)  / (nrow(cds %>% filter(learned_cats == i - 1)))
  el <- ellipse(cov, centre=matrix(as.numeric(cds_mus[i,])))
  graph <- graph + geom_point(data=as_tibble(el), aes(x=y, y=x))
}
graph + 
  scale_x_reverse() +
  scale_y_reverse()

