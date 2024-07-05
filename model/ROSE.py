import os
import pickle
import numpy as np
import pandas as pd

from .utils import validate_file

from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score

def get_features(path: str, k: int, size: int, bow):
    """Get features to be used in the train

    Parameters
    ----------
    path: str
        String with the path to the data
    k: int
        Number of the sample to be used.
    size: int
        Sample size to be used.
    """
    with open(path + f'/sparse_{size}_{k}.pkl', 'rb') as input_x:
        x = pickle.load(input_x)

    with open(path + f'/y_{size}_{k}.pkl', 'rb') as input_y:
        y = pickle.load(input_y)

    if bow:
        with open(path + f'/sparse_test_{size}_{k}.pkl', 'rb') as input_sparse:
            test_matrix = pickle.load(input_sparse)

        test_df = pd.read_csv(path + '/test.csv', delimiter=',')

        with open(path+f'/sparse_validation_{size}_{k}.pkl', 'rb') as input_sparse:
            validation_matrix = pickle.load(input_sparse)
        
        validate_df = pd.read_csv(path + f'/validation_{size}.csv', delimiter=',')
    
    else:
        with open(path + f'/sparse_test.pkl', 'rb') as input_sparse:
            test_matrix = pickle.load(input_sparse)

        test_df = pd.read_csv(path + '/test.csv', delimiter=',')

        with open(path+f'/sparse_validation_{size}.pkl', 'rb') as input_sparse:
            validation_matrix = pickle.load(input_sparse)
        
        validate_df = pd.read_csv(path + f'/validation_{size}.csv', delimiter=',')
    
    test_label = test_df.Label if bow else 1-test_df.Label
    validate_label = validate_df.Label if bow else 1-validate_df.Label

    return x, y, test_matrix, test_label, validation_matrix, validate_label

def rose_train(path, k, size, shrinkage, bow, jobs=11):
    x, y, test, y_test, validation, y_valid = get_features(path, k, size, bow)


    target_perct = 0.5
    # for target_perct in tqdm(np.arange(perct_minor, 0.51, 0.05)):
    file_not_exist, models_dict = validate_file(path,
                                                k,
                                                size,
                                                target_perct,
                                                f'ROSE_{shrinkage}')
    if file_not_exist:
        sample = int(y.sum() * (target_perct / (1 - target_perct)))
        print(f'Perct: {target_perct}')
        init_value = 0 if len(models_dict) == 0 else len(models_dict)

        for ind in tqdm(range(init_value, 40)):
            sm = RandomOverSampler(sampling_strategy={0: sample},
                                   shrinkage={0: shrinkage})
            x_sm, y_sm = sm.fit_resample(x, y)
            if bow:
                x_sm = np.round(x_sm).astype(int)

            regr = RandomForestClassifier(n_jobs=jobs, random_state= (ind+1) * (k+1))
            regr.fit(x_sm, y_sm)

            matrix = {}
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

            if not os.path.exists(path + f'/ROSE_{shrinkage}'):
                os.mkdir(path + f'/ROSE_{shrinkage}')
            
            file_path = path + f'/ROSE_{shrinkage}/target_{target_perct}_{size}_{k}.pkl'
            with open(file_path, 'wb') as dict_file:
                pickle.dump(models_dict, dict_file)
