import pickle
import pandas as pd
import logging

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 


def generate_data_tabular(path, df):
    perct_test = 1000/df.shape[0] 

    no_test, test = train_test_split(df, 
                                         test_size=perct_test,
                                         random_state=15)
    
    test.to_csv(path +'/test.csv')

    with open(path + f'/sparse_test.pkl', 'wb') as handle:
         pickle.dump(obj=csr_matrix(test.drop(columns=['Label'])), 
                     file=handle)

    sample_size = df.shape[0] -test.shape[0]

    for size in [500, 2000]:
            perct_validation = (size/4)/sample_size

            df_model, validation = train_test_split(no_test, 
                                           test_size=perct_validation,
                                           random_state=15)
            
            validation.to_csv(path + f'/validation_{size}.csv')

            with open(path + f'/sparse_validation_{size}.pkl', 'wb') as handle:
                pickle.dump(obj=validation.drop(columns=['Label']), 
                            file=handle)

            for k in range(50):
                train = df_model.sample(size, replace=False, random_state=k)
                train.to_csv(path + f'/train_{size}_{k}.csv')
                train_matrix = train.drop(columns=['Label'])
                y = 1-train['Label']

                with open(path + f'/sparse_{size}_{k}.pkl', 'wb') as handle:
                    pickle.dump(obj=train_matrix, file=handle)

                with open(path + f'/y_{size}_{k}.pkl', 'wb') as handle:
                    pickle.dump(obj=y, file=handle)


def load_default_credit(path):
    df = pd.read_csv('default_credit_dataset/UCI_Credit_Card.csv')
    df.drop(columns=['ID'], inplace=True)
    df.rename(columns={'default.payment.next.month': 'Label'}, inplace=True)
    df = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'],drop_first=True )
    generate_data_tabular(path, df)

def load_diabetes(path):
    # fetch dataset     
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    
    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 
    df = pd.concat([X, y], axis=1)
    df.rename(columns={'Diabetes_binary': 'Label'}, inplace=True)

    generate_data_tabular(path, df)

def load_marketing(path):
    # fetch dataset 
    df = pd.read_csv('digital_marketing_campaign_dataset.csv') 
    df.drop(columns=['CustomerID'])
    df = pd.get_dummies(df, columns=['Gender', 'CampaignChannel', 'CampaignType', 'AdvertisingPlatform', 'AdvertisingTool'], drop_first=True)
    df.rename(columns={'Conversion': 'Label'}, inplace=True)
    df['Label'] = 1-df['Label']
    generate_data_tabular(path, df)

def load_churn(path):
    df = pd.read_csv('Churn_Modelling.csv')
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    df.rename(columns={'Exited': 'Label'}, inplace=True)
    generate_data_tabular(path, df)


def break_full_df(df, path):
    test = pd.read_csv(path +'/test.csv')
    logging.info(f'Sample size: {df.shape[0]}')
    df = df[~df.index.isin(test['Unnamed: 0'])]
    logging.info(f'Sample size: {df.shape[0]}')
    perct_validation = 1/5
    df_model, validation = train_test_split(df,
                                        test_size=perct_validation,
                                        random_state=15)
    validation.to_csv(path + f'/validation_full.csv')
    with open(path + f'/sparse_validation_full.pkl', 'wb') as handle:
                pickle.dump(obj=validation.drop(columns=['Label']), 
                            file=handle)
    df_model.to_csv(path + '/train_full_0.csv')
    train_matrix = df_model.drop(columns=['Label'])
    with open(path + f'/sparse_full_0.pkl', 'wb') as handle:
                    pickle.dump(obj=train_matrix, file=handle)
    with open(path + '/y_full_0.pkl', 'wb') as handle:
        pickle.dump(obj=1-df_model.Label, file=handle)

def full_churn(path):
    df = pd.read_csv('Churn_Modelling.csv')
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    df.rename(columns={'Exited': 'Label'}, inplace=True)
    break_full_df(df, path)


def full_default_credit(path):
    df = pd.read_csv('default_credit_dataset/UCI_Credit_Card.csv')
    df.drop(columns=['ID'], inplace=True)
    df.rename(columns={'default.payment.next.month': 'Label'}, inplace=True)
    df = pd.get_dummies(df, columns=['EDUCATION', 'MARRIAGE'],drop_first=True )
    break_full_df(df, path)


def full_diabetes(path):
    # fetch dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    
    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 
    df = pd.concat([X, y], axis=1)
    df.rename(columns={'Diabetes_binary': 'Label'}, inplace=True)
    test = pd.read_csv(path +'/test.csv')
    logging.info(f'Sample size: {df.shape[0]}')
    df = df[~df.index.isin(test['Unnamed: 0'])]
    break_full_df(df, path)

def full_marketing(path):
    # fetch dataset 
    df = pd.read_csv('digital_marketing_campaign_dataset.csv') 
    df.drop(columns=['CustomerID'])
    df = pd.get_dummies(df, columns=['Gender', 'CampaignChannel', 'CampaignType', 'AdvertisingPlatform', 'AdvertisingTool'], drop_first=True)
    df.rename(columns={'Conversion': 'Label'}, inplace=True)
    df["Label"] = 1-df["Label"]
    break_full_df(df, path)
