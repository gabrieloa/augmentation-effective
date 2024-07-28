import os
import pickle
import pandas as pd

def get_roc_curves(data):
    df_list = []
    for k in range(len(data)):
        roc_point = []
        for thr in data[k]['matrix'].keys():
            tn, fp, fn, tp = data[k]['matrix'][thr].ravel()
            true_positive_rate = tp/(tp+fn)
            false_positive_rate = fp/(fp+tn)
            roc_point.append((false_positive_rate, true_positive_rate))
        df = pd.DataFrame(roc_point, columns=['FPR', 'TPR'])
        df['k'] = k
        df_list.append(df)
    df = pd.concat(df_list)
    df_pivot = df.pivot_table(index='FPR', columns='k', values='TPR').reset_index()
    return df_pivot

METHODS = ['Upsampling', 'SMOTE', 'BORDELINE', 'ADASYN']

def main():
    dir = os.listdir('./experiments')
    dir = [d for d in dir if d != 'logs']
    for d in dir:
        if not os.path.exists(f'./experiments/{d}/analysis'):
            os.mkdir(f'./experiments/{d}/analysis')
        for sample in [500, 2000]:
            for method in METHODS:
                print(f'Executando {d} {sample} {method}')
                file_save = f'./experiments/{d}/analysis/roc_curve_{method}_{sample}.csv'
                if os.path.exists(file_save):
                    continue
                else:
                    with open(f'./experiments/{d}/{method}/target_0.5_{sample}_0.pkl', 'rb') as file:
                        data = pickle.load(file)
                    df_tmp = get_roc_curves(data)
                    with open(f'./experiments/{d}/{method}/target_0.5_{sample}_1.pkl', 'rb') as file:
                        data = pickle.load(file)
                    df_tmp2 = get_roc_curves(data)
                    df_tmp = df_tmp.merge(df_tmp2, on='FPR', how='outer', suffixes=('_0', '_1')).sort_values('FPR')
                    for k in range(2, 50):
                        with open(f'./experiments/{d}/{method}/target_0.5_{sample}_{k}.pkl', 'rb') as file:
                            data = pickle.load(file)
                        df = get_roc_curves(data)
                        df_tmp = df_tmp.merge(df, on='FPR', how='outer', suffixes=('', f'_{k}')).sort_values('FPR')
                    
                    if 1 not in df_tmp['FPR']:
                        df_tmp = pd.concat(df_tmp, pd.DataFrame({'FPR': 1}))
                    df_tmp.loc[df_tmp.shape[0]-1, :] = 1
                    df_tmp = df_tmp.interpolate(method='linear', axis=0)
                    df_tmp.to_csv(file_save)
            
            file_save = f'./experiments/{d}/analysis/roc_curve_base_{sample}.csv'
            if os.path.exists(file_save):
                    continue
            else:
                with open(f'./experiments/{d}/metrics_base_{sample}_0.pkl', 'rb') as file:
                        data = pickle.load(file)
                df_tmp = get_roc_curves(data)
                with open(f'./experiments/{d}/metrics_base_{sample}_1.pkl', 'rb') as file:
                        data = pickle.load(file)
                df_tmp2 = get_roc_curves(data)
                df_tmp = df_tmp.merge(df_tmp2, on='FPR', how='outer', suffixes=('_0', '_1')).sort_values('FPR')
                for k in range(2, 50):
                    with open(f'./experiments/{d}/metrics_base_{sample}_{k}.pkl', 'rb') as file:
                        data = pickle.load(file)
                    df = get_roc_curves(data)
                    df_tmp = df_tmp.merge(df, on='FPR', how='outer', suffixes=('', f'_{k}')).sort_values('FPR')
                if 1 not in df_tmp['FPR']:
                    df_tmp = pd.concat(df_tmp, pd.DataFrame({'FPR': 1}))
                df_tmp.loc[df_tmp.shape[0]-1, :] = 1
                df_tmp.interpolate(method='linear', axis=0).to_csv(file_save)

if __name__ == '__main__':
    main()
