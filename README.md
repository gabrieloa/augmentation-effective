# augmentation-effective

Attention for the R scripts: Make sure that the path of the files is adjusted.

## Datasets
* Women Clothing Review Dataset: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
* Churn Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?select=customer_churn_dataset-training-master.csv
* Default Credit Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
* Marketing Dataset: https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset

## Section 1
To run the experiments to generate the image of the first section, run the following codes:

1. main_intro.py: 
    * Generate the samples for the app reviews dataset
    * Generate the base models
    * Generate the metrics for the 50 samples with different unbalanced percentages

2. analyses_intro/intro.ipynb 
    * Generate the metrics for the graphic
3. analyses_intro/intro.R
    * Generate the graphic

## Section 2
To generate the image of section 2, run the script:
1. analyses_intro/graph_prob.R
    * Generate the figure 2

## Section 3
### Main experiment
To execute the experiments with the datasets, execute the code:
1. main.py
    * Generate the samples for all datasets
    * Generate the base models
    * Generate the metrics for the 50 samples

### Analyses with Random Forest
To execute the test of the metrics, run the files:
2. analyses/generate_base_metrics.py:
    * Generate the confusion matrix of the base models.
3. analyses/test_*.ipynb
    * Generate the data of the hypothesis test for an specified metric
4. analyses/heatmap_*.R
    * Generate the heatmap of the test for a metric.

Replace the * with the metric of the figure.

To generate the functional boxplot analyses, execute:
5. analyses/generate_roc_curve.py:
    * Generate the roc curve data for the data
6. analyses/box_plot_functional.R
    * Generate the functional boxplot graphic

### Analyses with Logistic regression
To execute the test of the metrics, run the files:

2. analises_logistic/generate_base_metrics.py:
    * Generate the confusion matrix of the base models.
3. analises_logistic/test_ba.ipynb:
    * Generate the data of the hypothesis test for the balanced accuracy
4. analises/heatmap_ba.R:
    * Generate the heatmap of the test for the balanced accuracy.
5. analises_logistic/test_metrics.ipynb:
    * Generate the data for the hypothesis test for the Brier score and ROC AUC
6. analises/heatmap_metrics.R:
    * Generate the heatmap of the test for the Brier score and ROC AUC


### Experiment full
To execute the experiment with the full dataset, execute the following code:

1. main_full.py:
    * Generate the train and validation for all datasets.
    * Generate the base models
    * Generate the metrics for the 50 samples

2. analyses_full/generate_base_metrics.py:
    * Generate the confusion matrix of the base models.
3. analyses_full/test_ba.ipynb:
    * Generate the data of the hypothesis test for the balanced accuracy
4. analyses_full/heatmap_ba.R:
    * Generate the heatmap of the test for the balanced accuracy.
