import os
import time
import logging
import numpy as np
import argparse

from model.SMOTE import smote_train
from model.Upsampling import upsampling
from model.train_base_model import train_base_models
from load_data.from_hugging import *
from load_data.load_tabular import *

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


dataset = {'app': (os.path.join(experiment_folder, 'app_reviews'), load_app_review, None),
           'hatespeech': (os.path.join(experiment_folder, 'hatespeech'), load_hatespeech, None),
           'sentiment': (os.path.join(experiment_folder, 'sentiment'), load_sentiment, None),
           'women': (os.path.join(experiment_folder, 'womens'), load_women, None)}

dataset_tabular = {
                   'default_credit': (os.path.join(experiment_folder, 'default_credit'), load_default_credit),
                    'diabetes': (os.path.join(experiment_folder, 'diabetes'), load_diabetes),
                    'marketing': (os.path.join(experiment_folder, 'marketing'), load_marketing),
                    'churn': (os.path.join(experiment_folder, 'churn'), load_churn)
                   }

now = datetime.now()
if not os.path.exists(f'{experiment_folder}/logs'):
    os.mkdir(f'{experiment_folder}/logs')

file_name = f'{experiment_folder}/logs/experiments_{now.strftime("%d %m %Y %H:%M:%S")}.log'

if not os.path.exists(file_name):
    os.mknod(file_name)
logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO)

if not os.path.exists(file_name):
    os.mknod(file_name)
logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

paths = []
text_path = []
for key, value in dataset.items():
    path, load_data, perct = value
    paths.append(path)
    text_path.append(path)
    if not os.path.exists(path):
        os.mkdir(path)
        stars, review = load_data(data_folder=dataset_folder)
        dataset = DataSetHugginFace()
        dataset.generate_dataframe(stars, review, path, perct)

for key, value in dataset_tabular.items():
    path, load_data = value
    paths.append(path)
    if not os.path.exists(path):
        os.mkdir(path)
        load_data(path, dataset_folder)

sizes = [500, 2000]

for size in sizes:
    for path in paths:
        times = []
        logging.info(f'Path: {path}')
        for k in range(1):
            start_k = time.time()
            logging.info(f'Training baseline size: {size} k:{k}')
            start = time.time()
            train_base_models(path, size, k, 11)
            end = time.time()
            logging.info(
                f'End train baseline size: {size} k:{k} time:{end - start}')
            
            bow = True if path in text_path else False

            logging.info(f'Start upsamplig size:{size}, k:{k}')
            start = time.time()
            upsampling(path, k, size, bow, 11)
            end = time.time()
            logging.info(
                f'End upsamplig size:{size}, k:{k}, time:{end - start}')
            
            logging.info(f'Start upsamplig  Logistic size:{size}, k:{k}')
            start = time.time()
            upsampling(path, k, size, bow, 11, model_type='Logistic')
            end = time.time()
            logging.info(
                f'End upsamplig Logistic size:{size}, k:{k}, time:{end - start}')
            
            logging.info(f'Start SMOTE size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'SMOTE', bow, 11)
            end = time.time()
            logging.info(f'End SMOTE size:{size}, k:{k}, time:{end - start}')

            logging.info(f'Start SMOTE Logistic size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'SMOTE', bow, 11, model_type='Logistic')
            end = time.time()
            logging.info(f'End SMOTE Logistic size:{size}, k:{k}, time:{end - start}')


            logging.info(f'Start BORDELINE size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'BORDELINE', bow, 11)
            end = time.time()
            logging.info(
                f'End BORDELINE size:{size}, k:{k}, time:{end - start}')
            
            logging.info(f'Start BORDELINE Logistic size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'BORDELINE', bow, 11, model_type='Logistic')
            end = time.time()
            logging.info(
                f'End BORDELINE Logistic size:{size}, k:{k}, time:{end - start}')
            
            logging.info(f'Start ADASYN size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'ADASYN', bow, 11)
            end = time.time()
            logging.info(
                f'End ADASYN size:{size}, k:{k}, time:{end - start}')
            
            logging.info(f'Start ADASYN Logistic size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'ADASYN', bow, 11, model_type='Logistic')
            end = time.time()
            logging.info(
                f'End ADASYN Logistic size:{size}, k:{k}, time:{end - start}')
            
            end_k = time.time()
            logging.info(f'End k:{k} time:{end_k - start_k}')
            times.append(end_k - start_k)
            logging.info(f'Estimate time remaining: {round(np.median(times) * (50 - k) / 60, 2)} minutes')
            logging.info('-'*100)
        logging.info('-'*100)
    logging.info('#'*100)