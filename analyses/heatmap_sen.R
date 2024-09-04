require(ggplot2)

df <- read.csv('/home/Data/augmentation-effective/analyses/heatmap_sen.csv')

df['mean'] <- round(df['mean'])

df[is.na(df['mean_p']),'mean_p'] <- 0

facet_levels = c('c=0.5 for both model', 
                 'c=0.5 for augmented and \n optimized for base model',
                 'Optimized c for both model')


max(df['mean_p'])
min(df['mean_p'])

graphic <- ggplot(data = df, aes(x=method, y=dataset, fill=mean_p)) + 
  geom_tile()+
  geom_text(aes(label = mean_text), angle=60, size=5)+
  scale_fill_gradient2(limits=c(-95, 700), 
                       name='Percentage gain \n in Sensitivity')+
  theme(axis.text.x = element_text(angle = 45, hjust=1))+
  facet_grid(forcats::fct_rev(facet_b) ~ factor(facet_a, levels = facet_levels), scales = "free")+
  theme(axis.title = element_text(size = 21), 
        strip.text.x = element_text(size = 15),
        strip.text.y = element_text(size = 18),
        axis.text = element_text(size = 15),
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=15),
        legend.title = element_text(size=18),
        legend.text = element_text(size=15))
graphic
ggsave('/home/Data/augmentation-effective/analyses/sensitivity.pdf', 
       plot=graphic, 
       width = 12, 
       height = 10, 
       dpi = 300)


