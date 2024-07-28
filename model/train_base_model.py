import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_random_forest(x, y, n_jobs):
    regr = RandomForestClassifier(n_jobs=n_jobs)
    regr.fit(x, y)
    return regr


def validate_file(path, model_name, sample_size, k):
    list_files = os.listdir(path)
    num_files = len([1 for file in list_files if
                     file.startswith(f'{model_name}_{sample_size}_{k}')])
    return num_files


def read_features(path: str, sample_size: int, k: int):
    with open(path + f'/sparse_{sample_size}_{k}.pkl', 'rb') as input_matrix:
        x = pickle.load(input_matrix)

    with open(path + f'/y_{sample_size}_{k}.pkl', 'rb') as input_label:
        y = pickle.load(input_label)

    return x, y


def train_base_models(path: str, sample_size: int, k: int, n_jobs: int) -> None:
    """Train base line models of each experiment
    Parameters
    ----------
    path: str
        Path where the model files are in.
    sample_size: int
        Sample size used in the experiment.
    k: int
        Sample id.
    n_jobs: int
        Number of threads to be used in the Random Forest train.
    word2vec: dict
        Dictionary with the representation of each word.
    """
    trials = validate_file(path, 'rfbase', sample_size, k)
    if trials < 10:
        x, y = read_features(path, sample_size, k)
        for j in range(trials, 10):
            file = path + f'/rfbase_{sample_size}_{k}_rep{j}.pkl'
            regr = RandomForestClassifier(n_jobs=n_jobs, random_state=(j+1)*(k+1))
            regr.fit(x, y)
            with open(file, 'wb') as output_model:
                pickle.dump(regr, output_model)
            del regr

    trials = validate_file(path, 'logbase', sample_size, k)
    if trials < 10:
        x, y = read_features(path, sample_size, k)
        for j in range(trials, 10):
            file = path + f'/logbase_{sample_size}_{k}_rep{j}.pkl'
            model = LogisticRegression(random_state=(j+1)*(k+1))
            model.fit(x, y)
            with open(file, 'wb') as output_model:
                pickle.dump(model, output_model)
            del model
    del trials
