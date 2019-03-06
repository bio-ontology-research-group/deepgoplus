#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Input, Reshape,
    Flatten, BatchNormalization, Embedding,
    Conv1D, MaxPooling1D, Add, Concatenate)
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

MAXLEN = 1000

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--device', '-d', default='gpu:0',
    help='GPU or CPU device id')
@ck.option(
    '--model-file', '-mf', default='data/model_ipro_best.h5',
    help='Model filename')
@ck.option(
    '--batch-size', '-bs', default=256,
    help='Batch size')
@ck.option(
    '--data-file', '-df', default='data/data.pkl',
    help='Data file with sequences')
@ck.option(
    '--out-data-file', '-odf', default='data/data_features.pkl',
    help='Data file with sequences')
def main(device, model_file, batch_size, data_file, out_data_file):
    model = load_model(model_file)
    df = pd.read_pickle(data_file)
    generator = DFGenerator(df, {}, batch_size)
    with tf.device('/' + device):
        steps = math.ceil(len(df) / batch_size)
        preds = model.predict_generator(generator, steps=steps, verbose=1)
    df['features'] = list(preds)
    df.to_pickle(out_data_file)

class DFGenerator(object):

    def __init__(self, df, terms_dict, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.terms_dict = terms_dict
        self.nb_classes = len(terms_dict)
        
        
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
            data_onehot = np.zeros((len(df), MAXLEN, 21, 1), dtype=np.int32)
            
            for i, row in enumerate(df.itertuples()):
                onehot = to_onehot(row.sequences)
                data_onehot[i, :, :, 0] = onehot
                
                for t_id in row.interpros:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data_onehot, labels)
        else:
            self.reset()
            return self.next()

if __name__ == '__main__':
    main()
