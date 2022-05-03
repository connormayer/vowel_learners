library(tidyverse)

file <- 'C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/corpus_data/sp_ids_rev.csv'

foo <- read_csv(file)

ggplot(data=foo) + 
  geom_point(aes(x=f1, y=f2, color=vowel)) +
  facet_grid(~ register) + 
  ggtitle("Spanish")


file <- 'C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/corpus_data/feldman_learner_inputs.csv'

foo <- read_csv(file)

ggplot(data=foo) + 
  geom_point(aes(x=f1, y=f2, color=vowel)) +
  facet_grid(~ register) + 
  ggtitle("English")