import pickle
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from sklearn.utils import resample
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score

from .utils import validate_file

def get_features(path: str, k: int, size: int):
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

    with open(path + f'/y_{size}_{k}.pkl', 'rb') as input_y:
        y = pickle.load(input_y)

    with open(path + f'/tokenizer_{size}_{k}.pkl', 'rb') as input_token:
        tokenizer = pickle.load(input_token)

    train_df = pd.read_csv(path + f'/train_{size}_{k}.csv', delimiter=',')

    with open(path + f'/sparse_test_{size}_{k}.pkl', 'rb') as input_sparse:
        test_matrix = pickle.load(input_sparse)

    test_df = pd.read_csv(path + '/test.csv', delimiter=',')

    with open(path+f'/sparse_validation_{size}_{k}.pkl', 'rb') as input_sparse:
        validation_matrix = pickle.load(input_sparse)
    
    validate_df = pd.read_csv(path + f'/validation_{size}.csv', delimiter=',')

    return y, tokenizer, train_df, test_matrix, test_df.Label, validation_matrix, validate_df.Label


def get_features_tabular(path: str, k: int, size: int):
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

    with open(path + f'/y_{size}_{k}.pkl', 'rb') as input_y:
        y = pickle.load(input_y)

    train_df = pd.read_csv(path + f'/train_{size}_{k}.csv', delimiter=',').drop(columns=['Unnamed: 0'])

    with open(path + f'/sparse_test.pkl', 'rb') as input_sparse:
        test_matrix = pickle.load(input_sparse)

    test_df = pd.read_csv(path + '/test.csv', delimiter=',')

    with open(path+f'/sparse_validation_{size}.pkl', 'rb') as input_sparse:
        validation_matrix = pickle.load(input_sparse)

    validation_df = pd.read_csv(path + f'/validation_{size}.csv', delimiter=',')

    return y, train_df, test_matrix, 1-test_df.Label, validation_matrix, 1-validation_df.Label


def generate_new_train(train_df, label_column, label, samples, tokenizer, ind):
    upsample = resample(train_df[train_df[label_column] == label],
                        replace=True,
                        n_samples=samples, 
                        random_state=ind)
    dataset_upsample = pd.concat([train_df,
                                  upsample])
    train_sentence = dataset_upsample['text_process']
    y = np.asarray(dataset_upsample['Label'].to_list())

    train_matrix = csr_matrix(
        tokenizer.texts_to_matrix(train_sentence, 'count'))

    return train_matrix, y

def generate_new_train_tabular(train_df, label_column, label, samples, ind):
    upsample = resample(train_df[train_df[label_column] == 1-label],
                        replace=True,
                        n_samples=samples, 
                        random_state=ind)
    
    dataset_upsample = pd.concat([train_df,
                                  upsample])
    
    y = 1 - np.asarray(dataset_upsample['Label'].to_list())
    train_matrix = csr_matrix(dataset_upsample.drop(columns=['Label']))
    return train_matrix, y


def upsampling(path, k, size, bow, jobs=11):
    if bow:
        y, tokenizer, train_df, test, y_test, validation, y_valid = get_features(path, k, size)
    else:
        y, train_df, test, y_test, validation, y_valid = get_features_tabular(path, k, size)

    shape_y = y.shape[0]

    target_perct = .5
    file_not_exist, models_dict = validate_file(path,
                                                k,
                                                size,
                                                target_perct,
                                                'Upsampling')
    if file_not_exist:
        sample = int(y.sum() * (target_perct / (1 - target_perct)))
        sample = sample - shape_y + y.sum()

        init_value = 0 if len(models_dict) == 0 else len(models_dict)

        for ind in tqdm(range(init_value, 40)):
            if bow:
                x, y = generate_new_train(train_df,
                                        'Label',
                                        0,
                                        sample,
                                        tokenizer,
                                        ind)
            else:
                x, y = generate_new_train_tabular(train_df,
                                        'Label',
                                        0,
                                        sample,
                                        ind)
            
            regr = RandomForestClassifier(n_jobs=jobs, random_state= (ind+1) * (k+1))
            regr.fit(x, y)

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

            if not os.path.exists(path + f'/Upsampling'):
                os.mkdir(path + f'/Upsampling')
            
            file_path = path + f'/Upsampling/target_{target_perct}_{size}_{k}.pkl'
            with open(file_path, 'wb') as dict_file:
                pickle.dump(models_dict, dict_file)
