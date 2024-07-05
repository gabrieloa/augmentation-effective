import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .ProcessDf import process_df
from .GenerateFeatures import generate_features


def get_polarity(star):
        return 1 if star >= 3 else 0


def load_yelp():
    dataset = load_dataset('yelp_review_full')

    stars = dataset['train']['label'] + dataset['test']['label']

    review = dataset['train']['text'] + dataset['test']['text']

    polarity = [get_polarity(star) for star in stars]
    
    return polarity, review


def load_amazon():
    dataset = load_dataset('amazon_reviews_multi')

    stars = dataset['train']['stars']

    review = dataset['train']['review_body']
    id_language = [language == 'en' for language in
                   dataset['train']['language']]
    stars_filtered = [stars[ind] for ind in range(len(id_language)) if
                        id_language[ind]]
    review_filtered = [review[ind] for ind in range(len(id_language)) if
                        id_language[ind]]
    
    polarity = [get_polarity(star) for star in stars_filtered]
    
    return polarity, review_filtered


def load_app_review():
    dataset = load_dataset('app_reviews')

    stars = dataset['train']['star']

    review = dataset['train']['review']

    polarity = [get_polarity(star) for star in stars]

    return polarity, review


def load_hatespeech():
    dataset = load_dataset('hate_speech_offensive')

    label = dataset['train']['class']
    tweet = dataset['train']['tweet']
    non_neutral = [value in [0, 1] for value in label]

    label = [label[ind] for ind in range(len(label)) if
            non_neutral[ind]]
    tweet = [tweet[ind] for ind in range(len(tweet)) if non_neutral[ind]]

    return label, tweet


def load_sentiment():
    dataset = load_dataset('tweet_eval', 'sentiment')
    label = dataset['train']['label']
    tweet = dataset['train']['text']
    non_neutral = [value in [0, 2] for value in label]

    label = [1 if label[ind]==2 else 0 for ind in range(len(label)) if
            non_neutral[ind]]
    tweet = [tweet[ind] for ind in range(len(tweet)) if non_neutral[ind]]

    return label, tweet

def load_women():
    dataset = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
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
        
        # perct_validation = 1/5

        # df_model, validation = train_test_split(no_test, 
        #                                    test_size=perct_validation,
        #                                    random_state=15)
            
        # validation.to_csv(path + f'/validation.csv')
        # df_model.to_csv(path + '/train.csv')
        # generate_features(path, df_model, 0, 'full', test, validation)

