# augmentation-effective

## Datasets
* Women Clothing Review Dataset: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
* Churn Dataset: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?select=customer_churn_dataset-training-master.csv
* Default Credit Dataset: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
* Marketing Dataset: https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset

## Section 1

**Note: Before running the scripts in Section 1, ensure you have run the scripts in Section 3 first.**


To run the experiments and generate the image for the first section, execute the following scripts:

1. `main_intro.py`: 
    * Generates samples for the app reviews dataset.
    * Generates the base models.
    * Generates metrics for 50 samples with different unbalanced percentages.

    Usage:
    ```sh
    python main_intro.py --dataset_folder ./data --experiments_folder ./experiments
    ```

2. `analyses_intro/intro.py`:
    * Generates the metrics for the graphic.

    Usage:
    ```sh
    python analyses_intro/intro.py --experiments_folder ./experiments
    ```

3. `analyses_intro/intro.R`:
    * Generates the graphic.

    Usage:
    ```sh
    Rscript analyses_intro/intro.R
    ```

## Section 2
To generate the image for section 2, execute the script:

1. `analyses_intro/graph_prob.R`:
    * Generates Figure 2.

## Section 3
### Main experiment
To execute the experiments with the datasets, run the following script:

1. `main.py`:
    * Generates samples for all datasets.
    * Generates the base models.
    * Generates metrics for 50 samples.

    The main script has two parameters:
    * `--experiments_folder`: Folder where the experiments should be stored.
    * `--dataset_folder`: Folder containing the raw datasets.

    Example usage:

    ```sh
    python main.py --dataset_folder ./data --experiments_folder ./experiments
    ```

### Analyses with Random Forest
To execute the metric tests, run the following scripts:

2. `analyses/generate_base_metrics.py`:
    * Generates the confusion matrix for the base models.

    The script has one parameter:
    * `--experiments_folder`: Folder where the experiments have been stored.

    Example usage:

    ```sh
    python analyses/generate_base_metrics.py --experiments_folder ./experiments
    ```

3. `analyses/test_*.py`:
    * Generates the data for the hypothesis test for a specified metric.

    Example usage:

    ```sh
    python analyses/test_specificity.py --experiments_folder ./experiments
    ```
    
4. `analyses/heatmap_*.R`:
    * Generates the heatmap for the test of a metric.

    Example usage:

    ```sh
    Rscript analyses/heatmap_spe.R
    ```

Replace the `*` with the specific metric for the figure.

To generate the functional boxplot analyses, execute:

5. `analyses/generate_roc_curve.py`:
    * Generates the ROC curve data.

    Example usage:

    ```sh
    python analyses/generate_roc_curve.py --experiments_folder ./experiments
    ```

6. `analyses/box_plot_functional.R`:
    * Generates the functional boxplot graphic.

    The parameters to run this script are:
    * Path to the analyses folder.
    * Path to the experiments folder.
    * Name of the dataset (see the folder names in the experiments folder).
    * Size of the result: 500 or 2000.

    Example usage:

    ```sh
    Rscript analyses/box_plot_functional.R ./analyses ./experiments churn 500
    ```

### Analyses with Logistic regression
To execute the metric tests, run the following scripts:

2. `analyses_logistic/generate_base_metrics.py`:
    * Generates the confusion matrix for the base models.

    Example usage:

    ```sh
    python analyses_logistic/generate_base_metrics.py --experiments_folder ./experiments
    ```

3. `analyses_logistic/test_ba.py`:
    * Generates the data for the hypothesis test for balanced accuracy.

    Example usage:

    ```sh
    python analyses_logistic/test_ba.py --experiments_folder ./experiments
    ```

4. `analyses_logistic/heatmap_ba.R`:
    * Generates the heatmap for the test of balanced accuracy.

    Example usage:

    ```sh
    Rscript analyses_logistic/heatmap_ba.R
    ```

5. `analyses_logistic/test_metrics.py`:
    * Generates the data for the hypothesis test for the Brier score and ROC AUC.

    Example usage:

    ```sh
    python analyses_logistic/test_metrics.py --experiments_folder ./experiments
    ```

6. `analyses_logistic/heatmap_metrics.R`:
    * Generates the heatmap for the test of the Brier score and ROC AUC.

    Example usage:

    ```sh
    Rscript analyses_logistic/heatmap_metrics.R
    ```

### Experiment full
To execute the experiment with the full dataset, run the following script:

1. `main_full.py`:
    * Generates the train and validation sets for all datasets.
    * Generates the base models.
    * Generates metrics for 50 samples.

    Example usage:

    ```sh
    python main_full.py --dataset_folder ./data --experiments_folder ./experiments
    ```

2. `analyses_full/generate_base_metrics.py`:
    * Generates the confusion matrix for the base models.

    Example usage:

    ```sh
    python analyses_full/generate_base_metrics.py --experiments_folder ./experiments
    ```

3. 
3. `analyses_full/test_ba.py`:
    * Generates the data for the hypothesis test for balanced accuracy.

    Example usage:

    ```sh
    python analyses_full/test_ba.py --experiments_folder ./experiments
    ```

4. `analyses_full/heatmap_ba.R`:
    * Generates the heatmap for the test of balanced accuracy.

    Example usage:

    ```sh
    Rscript analyses_full/heatmap_ba.R
    ```
