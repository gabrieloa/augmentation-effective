require(ggplot2)


df_final <- read.csv("analyses_intro/boxplot.csv")

graphic <- ggplot(df_final, aes(x=as.factor(target_value), y=value)) + 
  geom_boxplot()+
  facet_grid(~variable)+
  geom_hline(yintercept = 0, color='red', linetype=2)+
  theme(axis.title = element_text(size = 24), 
        axis.text.x = element_text(size = 18),  # Ajusta o tamanho dos números no eixo x
        axis.text.y = element_text(size = 18),
        strip.text.y = element_text(size = 21),
        strip.text.x = element_text(size = 21),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))+
  labs(x="Ratio between minority and majority class proportions", 
       y="Percentage Gain in balanced accuracy \n from data augmentation")


ggsave('analyses_intro/intro.pdf', 
       plot=graphic, 
       width = 12, 
       height = 10, 
       dpi = 300)
