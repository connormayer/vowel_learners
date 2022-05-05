library(tidyverse)

setwd('C:/Users/conno/Dropbox/ling/vowel_learning_project/Code/vowel_learners/')
file <- 'training_data/hillenbrand_childes/hillenbrand_tidy_full.csv'

hb <- read_csv(file)

hb <- hb %>% 
  filter(F1 != 0 & F2 != 0)

my_plot <- ggplot(data=hb, aes(x=F2, y=F1, group=Vowel, color=Vowel, label=Vowel)) +
  scale_x_reverse() +
  scale_y_reverse() +
  stat_ellipse(lwd=3) +
  xlab("F2") +
  ylab("F1")

hb_ave <- hb %>% 
  group_by(Vowel) %>%
  summarize(F2=mean(F2), F1=mean(F1)) 

my_plot + 
  geom_label(data=hb_ave, aes(x=F2, y=F1, label=Vowel, color=Vowel)) +
  theme(legend.position="none")

ggsave("figures/hillenbrand_vowels.png")
