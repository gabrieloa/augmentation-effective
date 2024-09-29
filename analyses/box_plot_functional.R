require(fda)

args = commandArgs(trailingOnly=TRUE)

path = args[1]
experiment_path = args[2]
data = args[3]
size = args[4]

pdf(paste0(path, '/', data, '_', size, '.pdf'), 
    width = 12, height = 10)
par(mfrow=c(2, 2))

file_path <- paste0(experiment_path, '/', data, '/analysis/roc_curve_base_', size, '.csv')
df_base <- read.csv(file_path)
fpr_base <- df_base$FPR
tpr_base <- rowMeans(df_base[-c(1,2)])

file_path <- paste0(experiment_path, '/', data, '/analysis/roc_curve_Upsampling_', size, '.csv')
df_model <- read.csv(file_path)
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

file_path <- paste0(experiment_path, '/', data, '/analysis/roc_curve_SMOTE_', size, '.csv')
df_model <- read.csv(file_path)
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

file_path <- paste0(experiment_path, '/', data, '/analysis/roc_curve_BORDELINE_', size, '.csv')
df_model <- read.csv(file_path)
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

file_path <- paste0(experiment_path, '/', data, '/analysis/roc_curve_ADASYN_', size, '.csv')
df_model <- read.csv(file_path)
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


