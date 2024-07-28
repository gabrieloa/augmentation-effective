require(fda)
setwd('/home/Data/augmentation-effective/')

png('/home/Data/augmentation-effective/analises/churn_500.png', width = 12*300, height = 10*300, res = 300)
par(mfrow=c(2, 2))

df_base <- read.csv('/home/Data/augmentation-effective/experiments/churn/analysis/roc_curve_base_500.csv')
fpr_base <- df_base$FPR
tpr_base <- rowMeans(df_base[-c(1,2)])


df_model <- read.csv('/home/Data/augmentation-effective/experiments/churn/analysis/roc_curve_Upsampling_500.csv')
model_matrix <- as.matrix(df_model[-c(1,2)])
fpr <- df_model$FPR
fbplot(as.matrix(df_model[-c(1,2)]), 
       x=fpr, 
       method='MBD', 
       xlim = c(0,1), 
       ylim=c(0,1),
       xlab='False positive rate', 
       ylab='True positive rate', 
       main='Functional Boxplot ROC - Upsampling')
lines(fpr_base, tpr_base, col='black', lty=2, lwd=3)


df_model <- read.csv('/home/Data/augmentation-effective/experiments/churn/analysis/roc_curve_SMOTE_500.csv')
model_matrix <- as.matrix(df_model[-c(1,2)])
fpr <- df_model$FPR
fbplot(as.matrix(df_model[-c(1,2)]), 
       x=fpr, 
       method='MBD', 
       xlim = c(0,1), 
       ylim=c(0,1),
       xlab='False positive rate', 
       ylab='True positive rate', 
       main='Functional Boxplot ROC - SMOTE')
lines(fpr_base, tpr_base, col='black', lty=2, lwd=3)


df_model <- read.csv('/home/Data/augmentation-effective/experiments/churn/analysis/roc_curve_BORDELINE_500.csv')
model_matrix <- as.matrix(df_model[-c(1,2)])
fpr <- df_model$FPR
fbplot(as.matrix(df_model[-c(1,2)]), 
       x=fpr, 
       method='MBD', 
       xlim = c(0,1), 
       ylim=c(0,1),
       xlab='False positive rate', 
       ylab='True positive rate', 
       main='Functional Boxplot ROC - Bordeline')
lines(fpr_base, tpr_base, col='black', lty=2, lwd=3)

df_model <- read.csv('/home/Data/augmentation-effective/experiments/churn/analysis/roc_curve_ADASYN_500.csv')
model_matrix <- as.matrix(df_model[-c(1,2)])
fpr <- df_model$FPR
fbplot(as.matrix(df_model[-c(1,2)]), 
       x=fpr, 
       method='MBD', 
       xlim = c(0,1), 
       ylim=c(0,1),
       xlab='False positive rate', 
       ylab='True positive rate', 
       main='Functional Boxplot ROC - ADASYN')
lines(fpr_base, tpr_base, col='black', lty=2, lwd=3)
dev.off()


