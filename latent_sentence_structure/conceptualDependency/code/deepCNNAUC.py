from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras.utils import np_utils

import myCallbacks
import pdb


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# AUC for a binary classifier
def auc(y_true, y_pred):
    import tensorflow as tf
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    from keras import backend as K
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    from keras import backend as K
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P


MAX_WORDS = 12400
MAX_LENGTH = 2590 # maximum sequence length
EMBEDDING_DIM = 100

TRAIN_TEST_SPLIT = 0.2

if __name__=='__main__':
    courseData = pd.read_csv('data/newprepreqData.csv', sep=',')
    print (courseData.shape)
                    
    texts = []
    labels = []
                    
    for idx in range(courseData.textContent.shape[0]):
        text = BeautifulSoup(courseData.textContent[idx],"html5lib")
        cleanstring = text.get_text().strip().lower()
        cleanstring = cleanstring.encode( "utf-8",'ignore')
        texts.append(cleanstring)
        labels.append(courseData.prereqClass[idx])
        #print(labels)
                    
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
                    
    word_index = tokenizer.word_index
    #print('%s unique tokens were found.' % len(word_index))


    data = pad_sequences(sequences, maxlen=MAX_LENGTH)
                    
    labels = to_categorical(np.asarray(labels))
    print('Data (x) dimension:', data.shape)
    print('Label (y) dimension:', labels.shape)
                    
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    testdata_size = int(TRAIN_TEST_SPLIT * data.shape[0])
                    
    x_train = data[:-testdata_size]
    y_train = labels[:-testdata_size]
    x_test = data[-testdata_size:]
    y_test = labels[-testdata_size:]
                    
    print('y_test shape is', y_test.shape)
    testingvaldim = (x_test,y_test)
    print('valdata[1].shape',testingvaldim[1].shape)
    print('Number of positive and negative examples in trainging and test data ')
    print (y_train.sum(axis=0))
    print (y_test.sum(axis=0))
                    
    EMDED_DIR = "embeddings/glove"
    embeddings_index = {}

    f = open(os.path.join(EMDED_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        feature_values = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = feature_values
    f.close()


    #initialize embedding matrix randomly
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # for words in V with embeddings update their initialization with embedding vectors
            embedding_matrix[i] = embedding_vector
                    
    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_LENGTH, trainable=True)

    convolutions = []
    filter_windows = [3,5]


    sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

        #keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    for window in filter_windows:
        l_conv = Conv1D(nb_filter=128,filter_length=window,activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convolutions.append(l_pool)
                    
    l_merge = Merge(mode='concat', concat_axis=1)(convolutions)
    l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)
                    
    model = Model(sequence_input, preds)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    print("model fitting summary: CNN with multiple filters of different window sizes")
    model.summary()
    histories = myCallbacks.Histories()



    model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=20, batch_size=64,callbacks=[histories])


    print(histories.losses)
    print(histories.aucs)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


