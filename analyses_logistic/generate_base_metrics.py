import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix


for d in ['app_reviews', 'hatespeech', 'sentiment', 'womens']:
    for sample in [500, 2000]:
        y_valid = pd.read_csv(f'./experiments/{d}/validation_{sample}.csv')['Label']
        y_test = pd.read_csv(f'./experiments/{d}/test.csv')['Label']
        models_dict = {}
        for k in tqdm(range(50)):
            with open(f'./experiments/{d}/sparse_validation_{sample}_{k}.pkl', 'rb') as file:
                validation = pickle.load(file)

            with open(f'./experiments/{d}/sparse_test_{sample}_{k}.pkl', 'rb') as file:
                test = pickle.load(file)
            matrix = {}
            for ind in range(2):
                with open(f'./experiments/{d}/logbase_{sample}_{k}_rep{ind}.pkl', 'rb') as file:
                    regr = pickle.load(file)
                best_ba = 0
                best_thr = 0
                pred_valid = regr.predict_proba(validation)[:, 1]
                pred_test = regr.predict_proba(test)[:, 1]

                unique_prob = np.unique(np.round(np.concatenate((pred_valid, pred_test, np.array([.5]))), 3))
                unique_prob = np.sort(unique_prob)
                            
                for thr in unique_prob:
                    ba = balanced_accuracy_score(y_valid, pred_valid > thr)
                    if ba > best_ba:
                        best_ba = ba
                        best_thr = thr
                    matrix[thr] = confusion_matrix(y_test, pred_test > thr)

                    brier_score = np.mean((pred_test - y_test)**2)
                    auc = roc_auc_score(y_test, pred_test)


                models_dict[ind] = {'matrix': matrix,
                                                'thr': best_thr,
                                                'brier_score': brier_score,
                                                'auc': auc}
                with open(f'./experiments/{d}/metrics_base_log_{sample}_{k}.pkl', 'wb') as file:
                    pickle.dump(models_dict, file)

for d in ['default_credit', 'diabetes', 'churn', 'marketing']:
    test = pd.read_csv(f'./experiments/{d}/test.csv')
    y_test = 1-test['Label']
    test = test.drop(columns=['Label', 'Unnamed: 0'])
    for sample in [500, 2000]:
        valid = pd.read_csv(f'./experiments/{d}/validation_{sample}.csv')
        y_valid = 1-valid['Label']
        models_dict = {}
        validation = valid.drop(columns=['Label', 'Unnamed: 0'])   
        for k in tqdm(range(50)):
            matrix = {}
            for ind in range(2):
                with open(f'./experiments/{d}/logbase_{sample}_{k}_rep{ind}.pkl', 'rb') as file:
                    regr = pickle.load(file)
                best_ba = 0
                best_thr = 0
                pred_valid = regr.predict_proba(validation)[:, 1]
                pred_test = regr.predict_proba(test)[:, 1]

                unique_prob = np.unique(np.round(np.concatenate((pred_valid, pred_test, np.array([.5]))), 3))
                unique_prob = np.sort(unique_prob)
                            
                for thr in unique_prob:
                    ba = balanced_accuracy_score(y_valid, pred_valid > thr)
                    if ba > best_ba:
                        best_ba = ba
                        best_thr = thr
                    matrix[thr] = confusion_matrix(y_test, pred_test > thr)

                    brier_score = np.mean((pred_test - y_test)**2)
                    auc = roc_auc_score(y_test, pred_test)


                models_dict[ind] = {'matrix': matrix,
                                                'thr': best_thr,
                                                'brier_score': brier_score,
                                                'auc': auc}
                with open(f'./experiments/{d}/metrics_base_log_{sample}_{k}.pkl', 'wb') as file:
                    pickle.dump(models_dict, file)