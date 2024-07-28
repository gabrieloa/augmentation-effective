import os
import time
import logging
import numpy as np

from model.Upsampling import upsampling
from load_data.from_hugging import *

from datetime import datetime

dataset = {'app': ('./experiments/app_reviews/', load_app_review, None)}

now = datetime.now()
file_name = f'./experiments/logs/experiments_{now.strftime("%d %m %Y %H:%M:%S")}.log'

if not os.path.exists(file_name):
    os.mknod(file_name)
logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO)

if not os.path.exists(file_name):
    os.mknod(file_name)
logging.basicConfig(filename=file_name, filemode='w', level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

paths = ['./experiments/app_reviews/']
text_path = ['./experiments/app_reviews/']

sizes = [500]

for size in sizes:
    for path in paths:
        times = []
        logging.info(f'Path: {path}')
        for k in range(50):
            start_k = time.time()
            for target in np.arange(.25, .5, .05):
                target = np.round(target, 2)
                bow = True
                logging.info(f'Start upsamplig size:{size}, k:{k}, target:{target}')
                start = time.time()
                upsampling(path, k, size, bow, 11, target_perct=target)
                end = time.time()
                logging.info(f'End upsamplig size:{size}, k:{k}, target:{target}, time:{end - start}')

                end_k = time.time()
            logging.info(f'End k:{k} time:{end_k - start_k}')
            times.append(end_k - start_k)
            logging.info(f'Estimate time remaining: {round(np.median(times) * (50 - k) / 60, 2)} minutes')
            logging.info('-'*100)
        logging.info('-'*100)
    logging.info('#'*100)