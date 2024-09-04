require(fda)
setwd('/home/Data/augmentation-effective/')

pdf('/home/Data/augmentation-effective/analyses/churn_500.pdf', 
    width = 12, height = 10)
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
       main='Functional Boxplot ROC - Upsampling',
       cex.lab=1.4,
       cex.axis=1.2,
       alpha=1)
lines(fpr_base, tpr_base, col='black', lty=2, lwd=1)


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
       main='Functional Boxplot ROC - SMOTE',
       cex.lab=1.4,
       cex.axis=1.2,
       alpha=1)
lines(fpr_base, tpr_base, col='black', lty=2, lwd=1)


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
       main='Functional Boxplot ROC - Bordeline',
       cex.lab=1.4,
       cex.axis=1.2,
       alpha=1)
lines(fpr_base, tpr_base, col='black', lty=2, lwd=1)

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
       main='Functional Boxplot ROC - ADASYN',
       cex.lab=1.4,
       cex.axis=1.2,
       alpha=1)
lines(fpr_base, tpr_base, col='black', lty=2, lwd=1)
dev.off()


