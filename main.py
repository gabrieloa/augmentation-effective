import os
import time
import logging
import numpy as np

from model.ROSE import rose_train
from model.SMOTE import smote_train
from model.Upsampling import upsampling
from model.train_base_model import train_base_models
from load_data.from_hugging import *
from load_data.load_tabular import *

from datetime import datetime

dataset = {'app': ('./experiments/app_reviews/', load_app_review, None),
        #    'amazon': ('./experiments/amazon_reviews/', load_amazon, .1),
        #    'yelp': ('./experiments/yelp_reviews/', load_yelp, .1),
           'hatespeech': ('./experiments/hatespeech/', load_hatespeech, None),
           'sentiment': ('./experiments/sentiment/', load_sentiment, None),
           'women': ('./experiments/womens/', load_women, None)}

dataset_tabular = {
    # 'credit': ('./experiments/credit/', load_credit_card),
                   'default_credit': ('./experiments/default_credit/', load_default_credit),
                    'diabetes': ('./experiments/diabetes/', load_diabetes)
                   }

now = datetime.now()
file_name = f'./experiments/logs/experiments_{now.strftime("%d %m %Y %H:%M:%S")}.log'

if not os.path.exists(file_name):
    os.mknod(file_name)
logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO)

paths = []
text_path = []
for key, value in dataset.items():
    path, load_data, perct = value
    paths.append(path)
    text_path.append(path)
    if not os.path.exists(path):
        os.mkdir(path)
        stars, review = load_data()
        dataset = DataSetHugginFace()
        dataset.generate_dataframe(stars, review, path, perct)

for key, value in dataset_tabular.items():
    path, load_data = value
    paths.append(path)
    if not os.path.exists(path):
        os.mkdir(path)
        load_data(path)

sizes = [500, 2000]

for size in sizes:
    for path in paths:
        times = []
        logging.info(f'Path: {path}')
        for k in range(30, 40):
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
            upsampling(path, k, size, bow)
            end = time.time()
            logging.info(
                f'End upsamplig size:{size}, k:{k}, time:{end - start}')

            logging.info(f'Start SMOTE size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'SMOTE', bow)
            end = time.time()
            logging.info(f'End SMOTE size:{size}, k:{k}, time:{end - start}')

            logging.info(f'Start BORDELINE size:{size}, k:{k}')
            start = time.time()
            smote_train(path, k, size, 'BORDELINE', bow)
            end = time.time()
            logging.info(
                f'End BORDELINE size:{size}, k:{k}, time:{end - start}')
            
            for shrinkage in [0.5, 1, 3]:
                logging.info(f'Start ROSE {shrinkage} size:{size}, k:{k}')
                start = time.time()
                rose_train(path, k, size, shrinkage, bow)
                end = time.time()
                logging.info(
                    f'End ROSE {shrinkage} size:{size}, k:{k}, time:{end - start}')
            
            end_k = time.time()
            logging.info(f'End k:{k} time:{end_k - start_k}')
            times.append(end_k - start_k)
            logging.info(f'Estimate time remaining: {round(np.median(times) * (40 - k) / 60, 2)} minutes')
            logging.info('-'*100)
        logging.info('-'*100)
    logging.info('#'*100)