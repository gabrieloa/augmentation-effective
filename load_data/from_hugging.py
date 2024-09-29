import pandas as pd
import os

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .ProcessDf import process_df
from .GenerateFeatures import generate_features

import logging

def get_polarity(star):
        return 1 if star >= 3 else 0


def load_app_review(data_folder):
    dataset = load_dataset('app_reviews')

    stars = dataset['train']['star']

    review = dataset['train']['review']

    polarity = [get_polarity(star) for star in stars]

    return polarity, review


def load_hatespeech(data_folder):
    dataset = load_dataset('hate_speech_offensive')

    label = dataset['train']['class']
    tweet = dataset['train']['tweet']
    non_neutral = [value in [0, 1] for value in label]

    label = [label[ind] for ind in range(len(label)) if
            non_neutral[ind]]
    tweet = [tweet[ind] for ind in range(len(tweet)) if non_neutral[ind]]

    return label, tweet


def load_sentiment(data_folder):
    dataset = load_dataset('tweet_eval', 'sentiment')
    label = dataset['train']['label']
    tweet = dataset['train']['text']
    non_neutral = [value in [0, 2] for value in label]

    label = [1 if label[ind]==2 else 0 for ind in range(len(label)) if
            non_neutral[ind]]
    tweet = [tweet[ind] for ind in range(len(tweet)) if non_neutral[ind]]

    return label, tweet

def load_women(data_folder):
    if not os.path.exists(data_folder + '/Womens Clothing E-Commerce Reviews.csv'):
        raise FileNotFoundError('File Womens Clothing E-Commerce Reviews.csv not found')
    dataset = pd.read_csv(data_folder + '/Womens Clothing E-Commerce Reviews.csv')
    label = dataset['Recommended IND']
    text = dataset['Review Text']
    return label, text


class DataSetHugginFace():

    def generate_dataframe(self, polarity, review, path, perct=.1):
        
        tmp_df = pd.DataFrame({'review': review, 
                               'polarity': polarity})
        if perct is not None:
            samples = sum(polarity) * perct/(1-perct)
            df_negative = tmp_df[tmp_df.polarity == 0].sample(n=int(samples), 
                                                              random_state=15)

            df = pd.concat([tmp_df[tmp_df.polarity == 1], 
                            df_negative])
            del df_negative, tmp_df
        else:
            df = tmp_df
            del tmp_df
        
        df.to_parquet(path + '/data.parquet')

        df = process_df(df,
                        text_column='review', 
                        label_column='polarity')
        
        df = df[df.text_process!='null']
        df = df[df.text_process.str.strip()!='']
        
        perct_test = 1000/df.shape[0] 

        no_test, test = train_test_split(df, 
                                         test_size=perct_test,
                                         random_state=15)
        test.to_csv(path +'/test.csv')

        sample_size = df.shape[0] -test.shape[0]

        for size in [500, 2000]:
            perct_validation = (size/4)/sample_size

            df_model, validation = train_test_split(no_test, 
                                           test_size=perct_validation,
                                           random_state=15)
            
            validation.to_csv(path + f'/validation_{size}.csv')

            for k in range(50):
                train = df_model.sample(size, replace=False, random_state=k)
                train.to_csv(path + f'/train_{size}_{k}.csv')
                generate_features(path, train, k, size, test, validation)
        

def generate_full_sample(path):
    df = pd.read_parquet(path + '/data.parquet')
    df = process_df(df,
                    text_column='review', 
                    label_column='polarity')
    df = df[df.text_process!='null']
    df = df[df.text_process.str.strip()!='']

    test = pd.read_csv(path +'/test.csv')

    logging.info(f'Number of samples: {df.shape[0]}')
    df = df[~df.index.isin(test['Unnamed: 0'])]
    logging.info(f'Number of samples: {df.shape[0]}')
    perct_validation = 1/5
    df_model, validation = train_test_split(df, 
                                            test_size=perct_validation,
                                            random_state=15)
    validation.to_csv(path + f'/validation_full.csv')
    df_model.to_csv(path + '/train_full_0.csv')
    generate_features(path, df_model, 0, 'full', test, validation)
