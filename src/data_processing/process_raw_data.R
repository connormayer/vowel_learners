library(tidyverse)

get_distribution <- function(dataset) {
  
  dataset <- dataset %>%
    ungroup() %>%
    select(-c(speaker, language, register)) %>%
    group_by(vowel)
  
  mu <- dataset %>% 
    summarize(across(everything(), mean))
  
  cov_mat <- matrix(NA, nrow=nrow(mu), (ncol(dataset) - 1)^2)
  
  for (i in 1:nrow(mu)) {
    vals <- dataset %>% 
      ungroup() %>%
      filter(vowel == mu[i,]$vowel) %>%
      select(-vowel) %>%
      cov
    cov_mat[i, ] <- matrix(vals, nrow=1)
  }
  
  cov_tib <- as_tibble(cov_mat)
  cov_tib <- cov_tib %>% 
    mutate(vowel=mu$vowel) %>%
    relocate(vowel)
  
  cov_col_names <- 'vowel'
  for (i in 2:ncol(dataset)) {
    for (j in 2:ncol(dataset)) {
      cov_col_names <- c(cov_col_names, str_c(colnames(sp_train_ads_mu)[i], '-', colnames(sp_train_ads_mu)[j]))
    }
  }
  colnames(cov_tib) <- cov_col_names
  
  counts <- dataset %>% 
    group_by(vowel) %>%
    summarize(n=n())
  
  return(list(mu, cov_tib, counts))
}

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/')
file <- 'training_data/feldman_learner_inputs_barks.csv'

data <- read_csv(file)

sp <- data %>%
  filter(language == 'sp')

en <- data %>%
  filter(language == 'en')

# # Combine speakers
# en <- en %>%
#   mutate(speaker = str_replace(speaker, "\\d\\d$", ""))

# ggplot(data=sp, aes(x=f2, y=f1, color=vowel)) + 
#   geom_point(size=3) +
#   facet_grid(~ register) + 
#   scale_x_reverse() +
#   scale_y_reverse() +
#   ggtitle("Spanish")
# ggsave('figures/sp_with_outliers.png')

sp_no_outliers <- sp %>%
  group_by(speaker) %>%
  mutate(z_f1 = scale(f1), z_f2 = scale(f2)) %>%
  filter(between(z_f1, -2.5, +2.5) & between(z_f2, -2.5, +2.5)) %>%
  select(-contains("z_"))

# ggplot(data=sp_no_outliers, aes(x=f2, y=f1, color=vowel)) + 
#   geom_point(size=3) +
#   facet_grid(~ register) + 
#   scale_x_reverse() +
#   scale_y_reverse() +
#   ggtitle("Spanish")
# ggsave('figures/sp_no_outliers.png')

sp_ads <- sp_no_outliers %>%
  filter(register == 'ads')

# sp_ads %>% group_by(speaker) %>% count()

sp_train_ads <- sp_ads %>%
  filter(speaker %in% c('HS10', 'HS2', 'HS4'))


# sp_train %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

sp_test_ads <- sp_ads %>% 
  filter(speaker %in% c('HS6', 'HS8')) 

# sp_test %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

write_csv(sp_train_ads, 'training_data/spanish_ads_no_outliers_train_ads.csv')
write_csv(sp_test_ads, 'training_data/spanish_ads_no_outliers_test.csv')

result <- get_distribution(sp_train_ads)

write_csv(result[[1]], 'training_data/distributions/spanish_train_ads_mu.csv')
write_csv(result[[2]], 'training_data/distributions/spanish_train_ads_cov.csv')
write_csv(result[[3]], 'training_data/distributions/spanish_train_ads_counts.csv')

result <- get_distribution(sp_test_ads)

write_csv(result[[1]], 'training_data/distributions/spanish_test_ads_mu.csv')
write_csv(result[[2]], 'training_data/distributions/spanish_test_ads_cov.csv')
write_csv(result[[3]], 'training_data/distributions/spanish_test_ads_counts.csv')

sp_cds <- sp_no_outliers %>%
  select(-contains("z_")) %>%
  filter(register == 'cds')

write_csv(sp_cds, 'training_data/spanish_cds_no_outliers.csv')

result <- get_distribution(sp_cds)

write_csv(result[[1]], 'training_data/distributions/spanish_train_cds_mu.csv')
write_csv(result[[2]], 'training_data/distributions/spanish_train_cds_cov.csv')
write_csv(result[[3]], 'training_data/distributions/spanish_train_cds_counts.csv')

###########
# ENGLISH #
###########
# 
# ggplot(data=en, aes(x=f2, y=f1, color=vowel)) + 
#   geom_point(alpha=0.7, size=3) +
#   facet_grid(~ register) + 
#   scale_x_reverse() +
#   scale_y_reverse() +
#   ggtitle("English with outliers")
# ggsave('figures/en_with_outliers.png')

en_no_outliers <- en %>%
  group_by(speaker) %>%
  mutate(z_f1 = scale(f1), z_f2 = scale(f2)) %>%
  filter(between(z_f1, -2.5, +2.5) & between(z_f2, -2.5, +2.5)) %>%
  select(-contains("z_"))

# ggplot(data=en_no_outliers, aes(x=f2, y=f1, color=vowel)) + 
#   geom_point(size=3) +
#   facet_grid(~ register) + 
#   scale_x_reverse() +
#   scale_y_reverse() +
#   ggtitle("English")
# ggsave('figures/en_no_outliers.png')

en_ads <- en_no_outliers %>%
  filter(register == 'ads')

#en_ads %>% group_by(speaker) %>% count()

en_train_ads <- en_ads %>% 
  filter(speaker %in% c('s01', 's04', 's08'))

# en_train_ads %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

en_test_ads <- en_ads %>%
  filter(speaker %in% c('s09', 's12'))

# en_test %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

write_csv(en_train_ads, 'training_data/english_ads_no_outliers_train.csv')
write_csv(en_test_ads, 'training_data/english_ads_no_outliers_test.csv')

result <- get_distribution(en_train_ads)

write_csv(result[[1]], 'training_data/distributions/english_train_ads_mu.csv')
write_csv(result[[2]], 'training_data/distributions/english_train_ads_cov.csv')
write_csv(result[[3]], 'training_data/distributions/english_train_ads_counts.csv')

result <- get_distribution(en_test_ads)

write_csv(result[[1]], 'training_data/distributions/english_test_ads_mu.csv')
write_csv(result[[2]], 'training_data/distributions/english_test_ads_cov.csv')
write_csv(result[[3]], 'training_data/distributions/english_test_ads_counts.csv')

en_cds <- en_no_outliers %>%
  filter(register == 'cds')

# en_cds %>% group_by(speaker) %>% count()

write_csv(en_cds, 'training_data/english_cds_no_outliers.csv')

result <- get_distribution(en_cds)

write_csv(result[[1]], 'training_data/distributions/english_train_cds_mu.csv')
write_csv(result[[2]], 'training_data/distributions/english_train_cds_cov.csv')
write_csv(result[[3]], 'training_data/distributions/english_train_cds_counts.csv')
