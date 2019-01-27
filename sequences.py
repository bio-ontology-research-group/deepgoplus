#!/usr/bin/env python

"""
python nn_hierarchical_network.py
"""

import numpy as np
import pandas as pd
import click as ck
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Input, Reshape,
    Flatten, BatchNormalization, Embedding,
    Conv1D, MaxPooling1D, UpSampling1D)
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from collections import deque
import time
import logging
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math

from aminoacids import MAXLEN, to_ngrams, to_onehot

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--device',
    default='gpu:0',
    help='GPU or CPU device id')
@ck.option(
    '--model-file',
    default='data/model_seq.h5',
    help='Model filename')
@ck.option(
    '--batch-size',
    default=128,
    help='Batch size')
@ck.option(
    '--epochs',
    default=12,
    help='Training epochs')
@ck.option('--is-train', is_flag=True)
def main(device, model_file, batch_size, epochs, is_train):
    with tf.device('/' + device):
        train_df, valid_df, test_df = load_data()
        if is_train:
            train_model(train_df, valid_df, batch_size, epochs, model_file)
            
        test_model(test_df, batch_size, model_file)


def load_data(split=0.8, shuffle=True):
    df = pd.read_pickle('data/swissprot.pkl')
    # df = df[df['ngrams'].map(len) <= MAXLEN]
    n = len(df)
    index = np.arange(n)
    if shuffle:
        np.random.seed(seed=0)
        np.random.shuffle(index)
    train_n = int(n * split)
    valid_n = int(train_n * split)
    train_df = df.iloc[index[:valid_n]]
    valid_df = df.iloc[index[valid_n:train_n]]
    test_df = df.iloc[index[train_n:]]
    
    return train_df, valid_df, test_df


def get_model():
    embedding_dims = 128
    max_features = 8001
    input_seq = Input(shape=(MAXLEN,), dtype='float32', name='sequence')
    embed = Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN,
        name='seq_embed')(input_seq)
    conv = Conv1D(
        filters=128,
        kernel_size=8,
        padding='same',
        strides=1, name='seq_conv')(embed)
    print(conv.get_shape())
    pool = MaxPooling1D(pool_size=8, name='seq_pool')(conv)
    print(pool.get_shape())
    dec_conv1 = Conv1D(8, 16, padding='same', name='seq_conv2')(pool)
    print(dec_conv1.get_shape())
    dec_up = UpSampling1D(8, name='seq_up')(dec_conv1)
    print(dec_up.get_shape())
    decoded = Conv1D(21, 16, activation='sigmoid', padding='same', name='seq_conv3')(dec_up)
    print(decoded.get_shape())
    # decoded = Reshape((MAXLEN * 21,))(dec_conv2)
    # print(decoded.get_shape())
    
    model = Model(input_seq, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train_model(train_df, valid_df, batch_size, epochs, model_file):
    # set parameters:
    start_time = time.time()
    logging.info("Training data size: %d" % len(train_df))
    logging.info("Validation data size: %d" % len(valid_df))
    
    checkpointer = ModelCheckpoint(
        filepath=model_file,
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    logging.info('Starting training the model')

    train_generator = DFGenerator(train_df, batch_size)
    valid_generator = DFGenerator(valid_df, batch_size)
    valid_steps = int(math.ceil(len(valid_df) / batch_size))
    train_steps = int(math.ceil(len(train_df) / batch_size))
    model = get_model()
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_steps,
        max_queue_size=batch_size,
        workers=12,
        callbacks=[checkpointer, earlystopper])


def test_model(test_df, batch_size, model_file):
    test_generator = DFGenerator(test_df, batch_size)
    
    logging.info('Loading best model')
    model = load_model(model_file)
    
    logging.info('Predicting')
    test_steps = int(math.ceil(len(test_df) / batch_size))
    loss = model.evaluate_generator(
        test_generator, steps=test_steps, verbose=1)

    logging.info(f'Test loss {loss}')


class DFGenerator(object):

    def __init__(self, df, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        
        
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data_onehot = np.zeros((len(df), MAXLEN, 21), dtype=np.int32)
            data_seq = np.zeros((len(df), MAXLEN), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                seq = row.sequences[:(MAXLEN + 2)]
                ngrams = to_ngrams(seq)
                one_hot = to_onehot(seq)
                data_seq[i, 0:len(ngrams)] = ngrams
                data_onehot[i, :, :] = one_hot
                self.start += self.batch_size
            return (data_seq, data_onehot)
        else:
            self.reset()
            return self.next()


if __name__ == '__main__':
    main()
