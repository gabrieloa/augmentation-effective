require(ggplot2)

prob_aug <- seq(0, 1, 0.01) 

get_original <- function(p_y_1){
  p_a = 0.5
  prob_0 = (1-p_y_1)/p_a * (1-prob_aug)
  prob_1 = p_y_1/p_a * prob_aug
  return(prob_1/(prob_0+prob_1))
}

prob_01 = get_original(0.1)
prob_015 = get_original(0.15)
prob_02 = get_original(0.2)
prob_025 = get_original(0.25)

df=data.frame(prob_aug=prob_aug,prob_01=prob_01, prob_015=prob_015, 
              prob_02=prob_02, prob_025=prob_025)

graphic <- ggplot()+
  geom_line(aes(prob_01, prob_aug, color='0.1'))+
  geom_point(aes(0.1,0.5))+
  geom_line(aes(prob_015, prob_aug, color='0.15'))+
  geom_point(aes(0.15,0.5))+
  geom_line(aes(prob_02, prob_aug, color='0.2'))+
  geom_point(aes(0.2,0.5))+
  geom_line(aes(prob_025, prob_aug, color='0.25'))+
  geom_point(aes(0.25,0.5))+
  geom_abline(intercept=0, slope=1)+
  scale_colour_manual("P(Y=1)",
                      values = c("0.1" = "#EB5353", "0.15" = "#F9D923", 
                                 "0.2" = "#36AE7C", "0.25" = "#187498"))+
  labs(x=expression("P(Y=1|x)"), y=expression(paste("P"[a],"(Y=1|x)")))+
  geom_hline(yintercept=0.5, color='black', linetype=2)+
  geom_vline(xintercept=c(0.1, 0.15, 0.2, 0.25), color='black', linetype=2)+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))



ggsave('/home/Data/augmentation-effective/analises/example_prob.png', 
       plot=graphic, 
       width = 12, 
       height = 10, 
       dpi = 300)
