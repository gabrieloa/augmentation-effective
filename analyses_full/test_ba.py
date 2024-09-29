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

def generate_test(thr_bas_default, thr_aug_default):
    sample = 'full'
    k = 0
    results = []
    for d in dir:
        path_main = os.path.join(experiment_folder, d)
        for method in methods:
            pct_gain_list = []
            base_metrics = pickle.load(open(path_main + f'/metrics_base_{sample}_0.pkl', 'rb'))
            method_metrics = pickle.load(open(path_main + f'/{method}/target_0.5_{sample}_0.pkl', 'rb'))
            ba_list = []
            for k in range(len(base_metrics)):
                if thr_bas_default:
                    tn, fp, fn, tp = base_metrics[k]['matrix'][.5].ravel()
                else:
                    thr = base_metrics[k]['thr']
                    tn, fp, fn, tp = base_metrics[k]['matrix'][thr].ravel()
                ba = (tp/(tp+fn) + tn/(tn+fp))/2
                ba_list.append(ba)
            ba_model = []
            for k in range(len(method_metrics)):
                if thr_aug_default:
                    tn, fp, fn, tp = method_metrics[k]['matrix'][.5].ravel()
                else:
                    thr = method_metrics[k]['thr']
                    tn, fp, fn, tp = method_metrics[k]['matrix'][thr].ravel()
                ba = (tp/(tp+fn) + tn/(tn+fp))/2
                ba_model.append(ba)
            pct_gain = (np.array(ba_model) - np.mean(ba_list))/(np.mean(ba_list)) * 100
            pct_gain_list.append(pct_gain)
            pct_gain_list = np.array(pct_gain_list)
            sigma_j = pct_gain_list.std(axis=1)
            
            sample_pct_mean = []
            print('Começando bootstrap')
            for B in range(1000):
                sample_pct = []
                sample_pct_j = np.random.normal(0, sigma_j[0], 40)
                sample_pct.extend(sample_pct_j.tolist())
                sample_pct_mean.append(np.mean(sample_pct))
            p_value = np.mean(np.abs(sample_pct_mean) > np.abs(pct_gain_list.mean()))
            results.append([d, method, pct_gain_list.mean(), np.median(pct_gain_list), p_value])
    return results

list_df = []
results = generate_test(True, True)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

# Cria uma nova coluna 'mean_p' que é igual à coluna 'mean' onde 'p_value' < 0.01, e NaN caso contrário.
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'c=0.5 for both model'
df['facet_b'] = f'n=full'
list_df.append(df)

results = generate_test(False, True)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])
df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'c=0.5 for augmented and \n optimized for base model'
df['facet_b'] = f'n=full'
list_df.append(df)


results = generate_test(False, False)
df = pd.DataFrame(results, columns=['dataset', 'method', 'mean', 'median', 'p_value'])

df['mean'] = df['mean'].round()
df.loc[df['mean']==0, 'mean'] = 0
df['mean_p'] = np.where(df['p_value'] < 0.01, df['mean'], np.nan)
df['mean_text'] = np.where(df['p_value'] < 0.01, df['mean'].astype(int).astype(str), df['mean'].astype(int).apply(lambda x:f'{x}*'))
df['facet_a'] = 'Optimized c for both model'
df['facet_b'] = f'n=full'
list_df.append(df)

df_heatmap = pd.concat(list_df)
df_heatmap.to_csv('analyses_full/heatmap_ba.csv')