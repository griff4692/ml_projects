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
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras.utils import np_utils

#import myCallbacks
import pdb

K.set_image_dim_ordering('th')



MAX_WORDS = 13000
MAX_LENGTH = 100 # maximum sequence length
EMBEDDING_DIM = 100

#TRAIN_TEST_SPLIT = 0.2

if __name__=='__main__':
    #courseData = pd.read_csv('data/prereqDataBinary.csv', sep=',')
    #print (courseData.shape)
                    
    
    trainData = pd.read_csv('data/SNLItrain.csv',sep='|')
    testData = pd.read_csv('data/SNLItest.csv', sep='|')
    print (trainData.shape)
    print (testData.shape)

    texts = []
    labels = []
    
    def preprocess(string):
        #Tokenization/string cleaning for dataset
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        #string = re.sub(r"\"", "", string)
        return string.strip().lower()
                    
                    
    for idx in range(trainData.textContent.shape[0]):
        text = BeautifulSoup(trainData.textContent[idx])
        string = text.get_text().strip().lower()
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
       
        
        #cleanstring = cleanstring.encode( "utf-8",'ignore')
        texts.append(string)
        labels.append(trainData.prereqClass[idx])


    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    print('%s unique words.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print('Train Data (x) dimension:', data.shape)
    print('Train Label (y) dimension:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]


    #====== test data
    testtexts = []
    testlabels = []

    for testidx in range(testData.textContent.shape[0]):
        testtext = BeautifulSoup(testData.textContent[testidx])#,"html5lib")
        testcleanstring = testtext.get_text().strip().lower()
        string = re.sub(r"\\", "", testcleanstring)
        string = re.sub(r"\'", "", string)
        
        testtexts.append(string)
        testlabels.append(testData.prereqClass[testidx])


    testtokenizer = Tokenizer(num_words=MAX_WORDS)
    testtokenizer.fit_on_texts(testtexts)
    testsequences = tokenizer.texts_to_sequences(testtexts)

    testword_index = testtokenizer.word_index
    #print('%s unique tokens were found in test data.' % len(testword_index))


    testdata = pad_sequences(testsequences, maxlen=MAX_LENGTH)

    testlabels = to_categorical(np.asarray(testlabels))
    #print(labels)
    print('Test Data (x) dimension:', testdata.shape)
    print('Test Label (x) dimension:', testlabels.shape)

    testindices = np.arange(testdata.shape[0])
    np.random.shuffle(testindices)
    testdata = testdata[testindices]
    testlabels = testlabels[testindices]
    #================

    x_train = data
    y_train = labels
    x_test = testdata
    y_test = testlabels


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
    filter_windows = [3,4,5]


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
    l_dense = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(l_flat)
    preds = Dense(3, activation='softmax')(l_dense)
                    
    model = Model(sequence_input, preds)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("model fitting summary: CNN with multiple filters of different window sizes")
    model.summary()

    model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=20, batch_size=50)



