require(ggplot2)

df <- read.csv('/home/Data/augmentation-effective/analyses_logistic/heatmap_metrics.csv')

df['mean'] <- round(df['mean'])

df[is.na(df['mean_p']),'mean_p'] <- 0

facet_levels = c('AUC', 
                 'Brier Score')

graphic <- ggplot(data = df, aes(x=method, y=dataset, fill=mean_p)) + 
  geom_tile()+
  geom_text(aes(label = mean_text), angle=60, size=5)+
  scale_fill_gradient2(limits=c(-20, 15), 
                       name='Percentage gain')+
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
ggsave('/home/Data/augmentation-effective/analyses_logistic/heatmap_metrics_logistic.pdf', 
       plot=graphic, 
       width = 15, 
       height = 10, 
       dpi = 300)


