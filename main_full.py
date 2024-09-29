import os
import time
import logging
import numpy as np
import argparse

from model.SMOTE import smote_train
from model.Upsampling_full import upsampling
from model.train_base_model import train_base_models
from load_data.from_hugging import generate_full_sample
from load_data.load_tabular import full_default_credit, full_diabetes, full_marketing, full_churn

from datetime import datetime


parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--dataset_folder', type=str, required=True, help='Path to folder with raw dataset')
parser.add_argument('--experiments_folder', type=str, required=True, help='Path to folder where experiments will be saved')
args = parser.parse_args()
experiment_folder = args.experiments_folder
dataset_folder = args.dataset_folder

if not os.path.exists(dataset_folder):
    raise ValueError('Dataset folder does not exist')

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)


dataset = {'app': f'{experiment_folder}/app_reviews/',
           'hatespeech': f'{experiment_folder}/hatespeech/',
           'sentiment': f'{experiment_folder}/sentiment/',
           'women': f'{experiment_folder}/womens/'}

dataset_tabular = {
                    'diabetes': (f'{experiment_folder}/diabetes/'),
                   'default_credit': (f'{experiment_folder}/default_credit/'),
                    'marketing': (f'{experiment_folder}/marketing/'),
                    'churn': (f'{experiment_folder}/churn/')
                   }


now = datetime.now()
file_name = f'{experiment_folder}/logs/experiments_{now.strftime("%d %m %Y %H:%M:%S")}.log'

if not os.path.exists(file_name):
    os.mknod(file_name)
logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

paths = []
text_path = []
for key, value in dataset.items():
    path = value
    paths.append(path)
    text_path.append(path)
    logging.info(f'Generating full sample for {key}')
    generate_full_sample(path)
    

for key, value in dataset_tabular.items():
    path = value
    paths.append(path)
    logging.info(f'Generating full sample for {key}')
    if key == 'default_credit':
        full_default_credit(path, dataset_folder)
    elif key=='diabetes':
        full_diabetes(path, dataset_folder)
    elif key=='marketing':
        full_marketing(path, dataset_folder)
    else:
        full_churn(path, dataset_folder)

for path in paths:
    times = []
    logging.info(f'Path: {path}')
    k = 0
    size = 'full'
    logging.info(f'Training baseline size: {size} k:{k}')
    start = time.time()
    train_base_models(path, size, k, 11)
    end = time.time()
    logging.info(
                f'End train baseline size: {size} k:{k} time:{end - start}')
            
    bow = True if path in text_path else False

    logging.info(f'Start upsamplig size:{size}, k:{k}')
    start = time.time()
    upsampling(path, k, size, bow, 62)
    end = time.time()
    logging.info(
                f'End upsamplig size:{size}, k:{k}, time:{end - start}')

    logging.info(f'Start SMOTE size:{size}, k:{k}')
    start = time.time()
    smote_train(path, k, size, 'SMOTE', bow, 62)
    end = time.time()
    logging.info(f'End SMOTE size:{size}, k:{k}, time:{end - start}')

    logging.info(f'Start BORDELINE size:{size}, k:{k}')
    start = time.time()
    smote_train(path, k, size, 'BORDELINE', bow, 62)
    end = time.time()
    logging.info(f'End BORDELINE size:{size}, k:{k}, time:{end - start}')

    logging.info(f'Start ADASYN size:{size}, k:{k}')
    start = time.time()
    smote_train(path, k, size, 'ADASYN', bow, 62)
    end = time.time()
    logging.info(f'End ADASYN size:{size}, k:{k}, time:{end - start}')
            
    logging.info('-'*100)