library(assertr)
library(tidyverse)

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/')
file <- 'corpus_data/feldman_learner_inputs.csv'

data <- read_csv(file)

sp <- data %>%
  filter(language == 'sp')

en <- data %>%
  filter(language == 'en')

# Combine speakers
en <- en %>%
  mutate(speaker = str_replace(speaker, "\\d\\d$", ""))

ggplot(data=sp, aes(x=f2, y=f1, color=vowel)) + 
  geom_point(size=3) +
  facet_grid(~ register) + 
  scale_x_reverse() +
  scale_y_reverse() +
  ggtitle("Spanish")
ggsave('figures/sp_with_outliers.png')

sp_no_outliers <- sp %>%
  group_by(speaker) %>%
  mutate(z_f1 = scale(f1), z_f2 = scale(f2)) %>%
  filter(between(z_f1, -2.5, +2.5) & between(z_f2, -2.5, +2.5))

ggplot(data=sp_no_outliers, aes(x=f2, y=f1, color=vowel)) + 
  geom_point(size=3) +
  facet_grid(~ register) + 
  scale_x_reverse() +
  scale_y_reverse() +
  ggtitle("Spanish")
ggsave('figures/sp_no_outliers.png')

sp_ads <- sp_no_outliers %>%
  filter(register == 'ads')

sp_ads %>% group_by(speaker) %>% count()

sp_train <- sp_ads %>%
  filter(speaker %in% c('HS10', 'HS2', 'HS4'))

sp_train %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

sp_test <- sp_ads %>% 
  filter(speaker %in% c('HS6', 'HS8'))

sp_test %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

write_csv(sp_train, 'corpus_data/spanish_ads_no_outliers_train.csv')
write_csv(sp_test, 'corpus_data/spanish_ads_no_outliers_test.csv')

sp_cds <- sp_no_outliers %>%
  filter(register == 'ids')
write_csv(sp_cds, 'corpus_data/spanish_cds_no_outliers.csv')


sp_train_summary <- sp_train %>% 
  select(-contains("z_")) %>%
  group_by(vowel) %>% 
  summarize(n=n(), 
            mean_f1 = mean(f1),
            mean_f2 = mean(f2),
            mean_f3 = mean(f3),
            mean_duration = mean(duration),
            mean_df1_on = mean(df1_on),
            mean_df2_on = mean(df2_on),
            mean_df3_on = mean(df3_on),
            mean_df1_off = mean(df1_off),
            mean_df2_off = mean(df2_off),
            mean_df3_off = mean(df3_off),
            sd_f1 = sd(f1),
            sd_f2 = sd(f2),
            sd_f3 = sd(f3),
            sd_duration = sd(duration),
            sd_df1_on = sd(df1_on),
            sd_df2_on = sd(df2_on),
            sd_df3_on = sd(df3_on),
            sd_df1_off = sd(df1_off),
            sd_df2_off = sd(df2_off),
            sd_df3_off = sd(df3_off)
  )

write_csv(sp_train_summary, 'corpus_data/spanish_summary_train.csv')

sp_test_summary <- sp_test %>% 
  select(-contains("z_")) %>%
  group_by(vowel) %>% 
  summarize(n=n(), 
            mean_f1 = mean(f1),
            mean_f2 = mean(f2),
            mean_f3 = mean(f3),
            mean_duration = mean(duration),
            mean_df1_on = mean(df1_on),
            mean_df2_on = mean(df2_on),
            mean_df3_on = mean(df3_on),
            mean_df1_off = mean(df1_off),
            mean_df2_off = mean(df2_off),
            mean_df3_off = mean(df3_off),
            sd_f1 = sd(f1),
            sd_f2 = sd(f2),
            sd_f3 = sd(f3),
            sd_duration = sd(duration),
            sd_df1_on = sd(df1_on),
            sd_df2_on = sd(df2_on),
            sd_df3_on = sd(df3_on),
            sd_df1_off = sd(df1_off),
            sd_df2_off = sd(df2_off),
            sd_df3_off = sd(df3_off)
  )

write_csv(sp_test_summary, 'corpus_data/spanish_summary_test.csv')

###########
# ENGLISH #
###########

ggplot(data=en, aes(x=f2, y=f1, color=vowel)) + 
  geom_point(alpha=0.7, size=3) +
  facet_grid(~ register) + 
  scale_x_reverse() +
  scale_y_reverse() +
  ggtitle("English with outliers")
ggsave('figures/en_with_outliers.png')

en_no_outliers <- en %>%
  group_by(speaker) %>%
  mutate(z_f1 = scale(f1), z_f2 = scale(f2)) %>%
  filter(between(z_f1, -2.5, +2.5) & between(z_f2, -2.5, +2.5))

ggplot(data=en_no_outliers, aes(x=f2, y=f1, color=vowel)) + 
  geom_point(size=3) +
  facet_grid(~ register) + 
  scale_x_reverse() +
  scale_y_reverse() +
  ggtitle("English")
ggsave('figures/en_no_outliers.png')

en_ads <- en_no_outliers %>%
  filter(register == 'ads')

en_ads %>% group_by(speaker) %>% count()

en_train <- en_ads %>% 
  filter(speaker %in% c('s01', 's04', 's08'))

en_train %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

en_test <- en_ads %>%
  filter(speaker %in% c('s09', 's12'))

en_test %>% group_by(vowel) %>% summarize(n=n()) %>% mutate(freq=n/sum(n))

write_csv(en_train, 'corpus_data/english_ads_no_outliers_train.csv')
write_csv(en_test, 'corpus_data/english_ads_no_outliers_test.csv')

en_cds <- en_no_outliers %>%
  filter(register == 'ids')

en_cds %>% group_by(speaker) %>% count()

write_csv(en_cds, 'corpus_data/english_cds_no_outliers.csv')

en_train_summary <- en_train %>% 
  select(-contains("z_")) %>%
  group_by(vowel) %>% 
  summarize(n=n(), 
            mean_f1 = mean(f1),
            mean_f2 = mean(f2),
            mean_f3 = mean(f3),
            mean_duration = mean(duration),
            mean_df1_on = mean(df1_on),
            mean_df2_on = mean(df2_on),
            mean_df3_on = mean(df3_on),
            mean_df1_off = mean(df1_off),
            mean_df2_off = mean(df2_off),
            mean_df3_off = mean(df3_off),
            sd_f1 = sd(f1),
            sd_f2 = sd(f2),
            sd_f3 = sd(f3),
            sd_duration = sd(duration),
            sd_df1_on = sd(df1_on),
            sd_df2_on = sd(df2_on),
            sd_df3_on = sd(df3_on),
            sd_df1_off = sd(df1_off),
            sd_df2_off = sd(df2_off),
            sd_df3_off = sd(df3_off)
  )

write_csv(en_train_summary, 'corpus_data/english_summary_train.csv')

en_test_summary <- en_test %>% 
  select(-contains("z_")) %>%
  group_by(vowel) %>% 
  summarize(n=n(), 
            mean_f1 = mean(f1),
            mean_f2 = mean(f2),
            mean_f3 = mean(f3),
            mean_duration = mean(duration),
            mean_df1_on = mean(df1_on),
            mean_df2_on = mean(df2_on),
            mean_df3_on = mean(df3_on),
            mean_df1_off = mean(df1_off),
            mean_df2_off = mean(df2_off),
            mean_df3_off = mean(df3_off),
            sd_f1 = sd(f1),
            sd_f2 = sd(f2),
            sd_f3 = sd(f3),
            sd_duration = sd(duration),
            sd_df1_on = sd(df1_on),
            sd_df2_on = sd(df2_on),
            sd_df3_on = sd(df3_on),
            sd_df1_off = sd(df1_off),
            sd_df2_off = sd(df2_off),
            sd_df3_off = sd(df3_off)
  )

write_csv(en_test_summary, 'corpus_data/english_summary_test.csv')
