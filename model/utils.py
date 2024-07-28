import os
import pickle   

def validate_file(path, k, size, target_perct, model_type):
    file_path = path + f"/{model_type}/target_{target_perct}_{size}_{k}.pkl"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as dict_file:
            models_dict = pickle.load(dict_file)
        if len(models_dict) == 40:
            return False, {}
        else:
            return True, models_dict
    else:
        return True, {}


def convert_prob(y, prob):
    p_a = .5
    p = sum(y) / len(y)
    prob_1 = (p / p_a) * prob
    prob_0 = ((1 - p) / p_a) * (1 - prob)
    return prob_1 / (prob_1 + prob_0)