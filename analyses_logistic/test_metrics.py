import os
import numpy as np
import pickle
import pandas as pd 
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_folder', type=str, required=True, help='Path to folder where experiments will be saved')
args = parser.parse_args()
experiment_folder = args.experiments_folder

dir = [d for d in os.listdir(experiment_folder) if d!='logs']
methods = ['Upsampling_logistic', 'SMOTE_logistic', 'ADASYN_logistic', 'BORDELINE_logistic']

def generate_test_df(sample, metric='auc'):
    results = []
    for d in dir:
        path_main = os.path.join(experiment_folder, d)
        for method in methods:
            print(f'Metodo {method} em {d}')
            pct_gain_list = []
            for k in tqdm(range(50)):
                base_metrics = pickle.load(open(path_main + f'/metrics_base_log_{sample}_{k}.pkl', 'rb'))
                method_metrics = pickle.load(open(path_main + f'/{method}/target_0.5_{sample}_{k}.pkl', 'rb'))
                ba_list = []
                for k in range(len(base_metrics)):
                    br_score = base_metrics[k][metric]
                    ba_list.append(br_score)
                ba_model = []
                for k in range(len(method_metrics)):
                    br_score = method_metrics[k][metric]
                    ba_model.append(br_score)
                if metric=='brier_score':
                    pct_gain = (np.mean(ba_list) - np.array(ba_model))/(np.mean(ba_list)) * 100
                else:
                    pct_gain = (np.mean(ba_list) - np.array(ba_model))/(np.mean(ba_list)) * 100 * -1
                pct_gain_list.append(pct_gain)
            pct_gain_list = np.array(pct_gain_list)
            sigma_j = pct_gain_list.std(axis=1)
            sigma = pct_gain_list.mean(axis=1).std()

            sample_pct_mean = []
            print('Começando bootstrap')
            for B in range(1000):
                sample_pct = []
                sample_mu = np.random.normal(0, sigma, 50)
                for j in range(50):
                    sample_mu_j = sample_mu[j]
                    sample_pct_j = np.random.normal(sample_mu_j, sigma_j[j], 40)
                    sample_pct.extend(sample_pct_j.tolist())
                sample_pct_mean.append(np.mean(sample_pct))
            p_value = np.mean(np.abs(sample_pct_mean) > np.abs(pct_gain_list.mean()))
            results.append([d, method, pct_gain_list.mean(), np.median(pct_gain_list), p_value])
    return results

list_df = []
sample = 500
results = generate_test_df(sample, 'auc')
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

# Cria uma nova coluna 'mean_p' que é igual à coluna 'mean' onde 'p_value' < 0.01, e NaN caso contrário.
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'AUC'
df['facet_b'] = f'n={sample}'
list_df.append(df)


results = generate_test_df(sample, 'brier_score')
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

# Cria uma nova coluna 'mean_p' que é igual à coluna 'mean' onde 'p_value' < 0.01, e NaN caso contrário.
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'Brier Score'
df['facet_b'] = f'n={sample}'
list_df.append(df)

sample = 2000
results = generate_test_df(sample, 'auc')
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

# Cria uma nova coluna 'mean_p' que é igual à coluna 'mean' onde 'p_value' < 0.01, e NaN caso contrário.
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'AUC'
df['facet_b'] = f'n={sample}'
list_df.append(df)


results = generate_test_df(sample, 'brier_score')
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

# Cria uma nova coluna 'mean_p' que é igual à coluna 'mean' onde 'p_value' < 0.01, e NaN caso contrário.
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'Brier Score'
df['facet_b'] = f'n={sample}'
list_df.append(df)


df_heatmap = pd.concat(list_df)
df_heatmap["method"] = df_heatmap["method"].str.replace("_logistic", "")
df_heatmap.to_csv('analyses_logistic/heatmap_metrics.csv')