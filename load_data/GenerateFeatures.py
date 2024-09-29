import pickle
import numpy as np

from scipy.sparse import csr_matrix
from tensorflow.keras.preprocessing.text import Tokenizer


def generate_features(path, train, k, sample_size, test, validation):
    train_sentence = train['text_process']
    y = np.asarray(train['Label'].to_list())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentence)

    train_matrix = csr_matrix(
        tokenizer.texts_to_matrix(train_sentence, 'count'))

    with open(path + f'/tokenizer_{sample_size}_{k}.pkl', 'wb') as handle:
        pickle.dump(obj=tokenizer, file=handle)

    with open(path + f'/sparse_{sample_size}_{k}.pkl', 'wb') as handle:
        pickle.dump(obj=train_matrix, file=handle)

    with open(path + f'/y_{sample_size}_{k}.pkl', 'wb') as handle:
        pickle.dump(obj=y, file=handle)
    
    with open(path + f'/sparse_test_{sample_size}_{k}.pkl', 'wb') as handle:
        pickle.dump(obj=csr_matrix(tokenizer.texts_to_matrix(test['text_process'], 'count')), 
                    file=handle)

    with open(path + f'/sparse_validation_{sample_size}_{k}.pkl', 'wb') as handle:
        pickle.dump(obj=csr_matrix(tokenizer.texts_to_matrix(validation['text_process'], 'count')), 
                    file=handle)