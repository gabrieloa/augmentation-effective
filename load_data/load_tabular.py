import pickle
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 


def load_default_credit(path):
    df = pd.read_csv('default_credit_dataset/UCI_Credit_Card.csv')
    df.drop(columns=['ID'], inplace=True)
    df.rename(columns={'default.payment.next.month': 'Label'}, inplace=True)

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
                pickle.dump(obj=csr_matrix(validation.drop(columns=['Label'])), 
                            file=handle)

            for k in range(50):
                train = df_model.sample(size, replace=False, random_state=k)
                train.to_csv(path + f'/train_{size}_{k}.csv')
                train_matrix = csr_matrix(train.drop(columns=['Label']))
                y = 1-train['Label']

                with open(path + f'/sparse_{size}_{k}.pkl', 'wb') as handle:
                    pickle.dump(obj=train_matrix, file=handle)

                with open(path + f'/y_{size}_{k}.pkl', 'wb') as handle:
                    pickle.dump(obj=y, file=handle)
    
    # perct_validation = 1/5
    # df_model, validation = train_test_split(no_test, 
    #                                        test_size=perct_validation,
    #                                        random_state=15)
            
    # validation.to_csv(path + f'/validation.csv')
    # with open(path + f'/sparse_validation_full.pkl', 'wb') as handle:
    #             pickle.dump(obj=csr_matrix(validation.drop(columns=['Label'])), 
    #                         file=handle)
    # with open(path + '/sparse_full.pkl', 'wb') as handle:
    #             pickle.dump(obj=csr_matrix(df_model.drop(columns=['Label'])), 
    #                         file=handle)
    # y = 1-df_model['Label']
    # with open(path + '/y_full.pkl', 'wb') as handle:
    #             pickle.dump(obj=y, file=handle)
    
                
                


def load_diabetes(path):
    # fetch dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    
    # data (as pandas dataframes) 
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 
    df = pd.concat([X, y], axis=1)
    df.rename(columns={'Diabetes_binary': 'Label'}, inplace=True)

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
                pickle.dump(obj=csr_matrix(validation.drop(columns=['Label'])), 
                            file=handle)

            for k in range(50):
                train = df_model.sample(size, replace=False, random_state=k)
                train.to_csv(path + f'/train_{size}_{k}.csv')
                train_matrix = csr_matrix(train.drop(columns=['Label']))
                y = 1-train['Label']

                with open(path + f'/sparse_{size}_{k}.pkl', 'wb') as handle:
                    pickle.dump(obj=train_matrix, file=handle)

                with open(path + f'/y_{size}_{k}.pkl', 'wb') as handle:
                    pickle.dump(obj=y, file=handle)

    # perct_validation = 1/5
    # df_model, validation = train_test_split(no_test, 
    #                                        test_size=perct_validation,
    #                                        random_state=15)
            
    # validation.to_csv(path + f'/validation.csv')
    # with open(path + f'/sparse_validation_full.pkl', 'wb') as handle:
    #             pickle.dump(obj=csr_matrix(validation.drop(columns=['Label'])), 
    #                         file=handle)
    # with open(path + '/sparse_full.pkl', 'wb') as handle:
    #             pickle.dump(obj=csr_matrix(df_model.drop(columns=['Label'])), 
    #                         file=handle)
    # y = 1-df_model['Label']
    # with open(path + '/y_full.pkl', 'wb') as handle:
    #             pickle.dump(obj=y, file=handle)