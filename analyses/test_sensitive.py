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
methods = ['Upsampling', 'SMOTE', 'ADASYN', 'BORDELINE']

def generate_test(sample, thr_bas_default, thr_aug_default):
    results = []
    for d in dir:
        path_main = os.path.join(experiment_folder, d)
        for method in methods:
            pct_gain_list = []
            for k in tqdm(range(50)):
                base_metrics = pickle.load(open(path_main + f'/metrics_base_{sample}_{k}.pkl', 'rb'))
                method_metrics = pickle.load(open(path_main + f'/{method}/target_0.5_{sample}_{k}.pkl', 'rb'))
                sen_lis = []
                for k in range(len(base_metrics)):
                    if thr_bas_default:
                        tn, fp, fn, tp = base_metrics[k]['matrix'][.5].ravel()
                    else:
                        thr = base_metrics[k]['thr']
                        tn, fp, fn, tp = base_metrics[k]['matrix'][thr].ravel()
                    sen = (tn)/(tn+fp)
                    sen_lis.append(sen)
                sen_model = []
                for k in range(len(method_metrics)):
                    if thr_aug_default:
                        tn, fp, fn, tp = method_metrics[k]['matrix'][.5].ravel()
                    else:
                        thr = method_metrics[k]['thr']
                        tn, fp, fn, tp = method_metrics[k]['matrix'][thr].ravel()
                    sen = (tn)/(tn+fp)
                    sen_model.append(sen)
                if np.mean(sen_lis) == 0:
                    continue
                pct_gain = (np.array(sen_model) - np.mean(sen_lis))/(np.mean(sen_lis)) * 100
                pct_gain_list.append(pct_gain)
            pct_gain_list = np.array(pct_gain_list)
            mu_j = pct_gain_list.mean(axis=1)
            sigma_j = pct_gain_list.std(axis=1)
            sigma = pct_gain_list.mean(axis=1).std()

            sample_pct_mean = []
            print('Começando bootstrap')
            for B in range(1000):
                sample_pct = []
                sample_mu = np.random.normal(0, sigma, len(mu_j))
                for j in range(len(mu_j)):
                    sample_mu_j = sample_mu[j]
                    sample_pct_j = np.random.normal(sample_mu_j, sigma_j[j], 40)
                    sample_pct.extend(sample_pct_j.tolist())
                sample_pct_mean.append(np.mean(sample_pct))
            p_value = np.mean(np.abs(sample_pct_mean) > np.abs(pct_gain_list.mean()))
            results.append([d, method, pct_gain_list.mean(), np.median(pct_gain_list), p_value])
    return results

list_df = []
sample = 500
results = generate_test(sample, True, True)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'c=0.5 for both model'
df['facet_b'] = f'n={sample}'
list_df.append(df)

results = generate_test(sample, False, True)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'c=0.5 for augmented and \n optimized for base model'
df['facet_b'] = f'n={sample}'
list_df.append(df)


results = generate_test(sample, False, False)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'Optimized c for both model'
df['facet_b'] = f'n={sample}'
list_df.append(df)


sample = 2000
results = generate_test(sample, True, True)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'c=0.5 for both model'
df['facet_b'] = f'n={sample}'
list_df.append(df)

results = generate_test(sample, False, True)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'c=0.5 for augmented and \n optimized for base model'
df['facet_b'] = f'n={sample}'
list_df.append(df)


results = generate_test(sample, False, False)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'Optimized c for both model'
df['facet_b'] = f'n={sample}'
list_df.append(df)

df_heatmap = pd.concat(list_df)
df_heatmap.to_csv('analyses/heatmap_sen.csv')