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

list_df = []
for idx in range(50):
    with open(f"{experiment_folder}/app_reviews/metrics_base_500_{idx}.pkl", "rb") as f:
        base = pickle.load(f)
    ba_base = []
    ba_opt = []
    for k in range(len(base)):
        tn, fp, fn, tp = base[k]['matrix'][.5].ravel()
        ba = (tp/(tp+fn) + tn/(tn+fp))/2
        ba_base.append(ba)
        thr = base[k]['thr']
        tn, fp, fn, tp = base[k]['matrix'][thr].ravel()
        ba = (tp/(tp+fn) + tn/(tn+fp))/2
        ba_opt.append(ba)
    for target in np.arange(.25, .51, .05):
        target = np.round(target, 2)
        with open(f"{experiment_folder}/app_reviews/Upsampling/target_{target}_500_{idx}.pkl", "rb") as f:
            target_dict = pickle.load(f)
        ba_target = []
        ba_target_opt = []
        for k in range(len(target_dict)):
            tn, fp, fn, tp = target_dict[k]['matrix'][.5].ravel()
            ba = (tp/(tp+fn) + tn/(tn+fp))/2
            ba_target.append(ba)
            thr = target_dict[k]['thr']
            tn, fp, fn, tp = target_dict[k]['matrix'][thr].ravel()
            ba = (tp/(tp+fn) + tn/(tn+fp))/2
            ba_target_opt.append(ba)
        df_tmp = pd.DataFrame({'target': ba_target, 'target_opt': ba_target_opt})
        df_tmp["target_value"] = target
        df_tmp["base"] = np.mean(ba_base)
        df_tmp["base_opt"] = np.mean(ba_opt)
        list_df.append(df_tmp)        

df_final = pd.concat(list_df)
df_final["Cut-off 0.5 (default)"] = (df_final.target - df_final.base)/df_final.base * 100
df_final["Optimized cut-off"] = (df_final.target - df_final.base_opt)/df_final.base_opt * 100
df_final = df_final.melt(id_vars=["target_value"], value_vars=["Cut-off 0.5 (default)", "Optimized cut-off"])
df_final.to_csv("analyses_intro/boxplot.csv")