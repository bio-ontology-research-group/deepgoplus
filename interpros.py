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

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--device',
    default='gpu:0',
    help='GPU or CPU device id')
@ck.option(
    '--model-file',
    default='model.h5',
    help='Model filename')
@ck.option(
    '--batch-size',
    default=64,
    help='Batch size')
@ck.option(
    '--epochs',
    default=12,
    help='Training epochs')
@ck.option(
    '--model-file',
    default='data/model_ipro.h5',
    help='Model filename')
@ck.option('--is-train', is_flag=True)
def main(device, model_file, batch_size, epochs, is_train):
    global interpros
    df = pd.read_pickle('data/interpros.pkl')
    interpros = df['interpros'].values
    global nb_classes
    nb_classes = len(interpros)
    global interpro_ix
    interpro_ix = {}
    for i, ipro in enumerate(interpros):
        interpro_ix[ipro] = i

    with tf.device('/' + device):
        get_model()
        train_df, valid_df, test_df = load_data()
        if is_train:
            train_model(train_df, valid_df, batch_size, epochs, model_file)
            
        test_model(test_df, batch_size, model_file)

def load_data(split=0.8, shuffle=True):
    df = pd.read_pickle('data/swissprot_exp.pkl')
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
    
    # augment training data
    def augment_df(df, n):
        starts = list()
        reverse = list()
        seqids = list()
        sequences = {}
        ipros = {}
        for i, row in enumerate(df.itertuples()):
            sequences[i] = row.sequences
            ipros[i] = row.interpros
            seq_len = (row.sequences)
            m = min(n, max(1, MAXLEN - len(row.sequences)))
            for j in range(m):
                start = np.random.choice(max(1, MAXLEN - len(row.sequences)))
                rev = np.random.choice([True, False])
                starts.append(start)
                reverse.append(rev)
                seqids.append(i)
            if n == 0:
                starts.append(0)
                reverse.append(False)
                seqids.append(i)

        starts = np.array(starts)
        reverse = np.array(reverse)
        seqids = np.array(seqids)
        # Shuffle
        index = np.arange(len(seqids))
        np.random.seed(seed=0)
        np.random.shuffle(index)
        starts = starts[index]
        reverse = reverse[index]
        seqids = seqids[index]
        
        return sequences, ipros, starts, reverse, seqids

    train = augment_df(train_df, 10)
    valid = augment_df(valid_df, 0)
    test = augment_df(test_df, 10)
    
    return train, valid, test


def get_model():
    input_seq = Input(shape=(MAXLEN, 21), dtype='float32', name='sequence')
    kernels = list(range(8, 129, 8))
    nets = []
    for i in range(len(kernels)):
        conv = Conv1D(
            filters=32,
            kernel_size=kernels[i],
            padding='valid',
            name='ipro_conv_' + str(i),
            activation='relu',
            kernel_initializer='glorot_normal')(input_seq)
        print(conv.get_shape())
        pool = MaxPooling1D(
            pool_size=MAXLEN - kernels[i] + 1, name='ipro_pool_' + str(i))(conv)
        flat = Flatten(name='ipro_flat_' + str(i))(pool)
        nets.append(flat)
    net = Concatenate(axis=1)(nets)
    output = Dense(nb_classes, activation='sigmoid', name='ipro_out')(net)
    
    model = Model(input_seq, output)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
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

    train_generator = DFGenerator(train_df, interpro_ix, batch_size)
    valid_generator = DFGenerator(valid_df, interpro_ix, batch_size)
    valid_steps = int(math.ceil(valid_generator.size / batch_size))
    train_steps = int(math.ceil(train_generator.size / batch_size))
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
    test_generator = DFGenerator(test_df, interpro_ix, batch_size)
    sequences, ipros, starts, reverse, seqids = test_df
        
    logging.info('Loading best model')
    model = load_model(model_file)
    
    logging.info('Predicting')
    test_steps = int(math.ceil(len(seqids) / batch_size))
    preds = model.predict_generator(
        test_generator, steps=test_steps, verbose=1)

    logging.info('Computing performance')
    test_labels = np.zeros((len(seqids), nb_classes), dtype=np.int32)
    for i, j in enumerate(seqids):
        for ipro in ipros[j]:
            if ipro in interpro_ix:
                test_labels[i, interpro_ix[ipro]] = 1
    
    f, p, r, t, preds_max = compute_performance(preds, test_labels)
    roc_auc = compute_roc(preds, test_labels)
    mcc = compute_mcc(preds_max, test_labels)
    logging.info('Fmax measure: \t %f %f %f %f' % (f, p, r, t))
    logging.info('ROC AUC: \t %f ' % (roc_auc, ))
    logging.info('MCC: \t %f ' % (mcc, ))
    print('%.3f & %.3f & %.3f & %.3f & %.3f' % (
        f, p, r, roc_auc, mcc))
    
def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        total = 0
        f = 0.0
        p = 0.0
        r = 0.0
        p_total = 0
        for i in range(labels.shape[0]):
            tp = np.sum(predictions[i, :] * labels[i, :])
            fp = np.sum(predictions[i, :]) - tp
            fn = np.sum(labels[i, :]) - tp
            # all_gos = set()
            # for go_id in gos[i]:
            #     if go_id in all_functions:
            #         all_gos |= get_anchestors(go, go_id)
            # all_gos.discard(GO_ID)
            # all_gos -= func_set
            # fn += len(all_gos)
            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall
        if p_total == 0:
            continue
        r /= total
        p /= p_total
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f_max < f:
                f_max = f
                p_max = p
                r_max = r
                t_max = threshold
                predictions_max = predictions
    return f_max, p_max, r_max, t_max, predictions_max


class DFGenerator(object):

    def __init__(self, df, terms_dict, batch_size):
        sequences, ipros, starts, reverse, seqids = df
        self.sequences = sequences
        self.ipros = ipros
        self.starts = starts
        self.reverse = reverse
        self.seqids = seqids
        
        self.start = 0
        self.size = len(seqids)
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
            seqids = self.seqids[batch_index]
            starts = self.starts[batch_index]
            reverse = self.reverse[batch_index]
            labels = np.zeros((len(seqids), self.nb_classes), dtype=np.int32)
            data_onehot = np.zeros((len(seqids), MAXLEN, 21), dtype=np.int32)
            
            for i in range(len(seqids)):
                if reverse[i]:
                    revseq = self.sequences[seqids[i]][::-1]
                    onehot = to_onehot(revseq, start=starts[i])
                else:
                    onehot = to_onehot(self.sequences[seqids[i]], start=starts[i])
                data_onehot[i, :, :] = onehot
                for t_id in self.ipros[seqids[i]]:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data_onehot, labels)
        else:
            self.reset()
            return self.next()


if __name__ == '__main__':
    main()
