#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import math

from utils import GeneOntology, read_fasta
from aminoacids import MAXLEN, to_ngrams

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, MaxPooling1D, Flatten, Dot, Activation
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/string/data.pkl',
    help='StringDB interactions data')
@ck.option(
    '--proteins-file', '-pf',
    default='data/string/proteins.pkl',
    help='Protein sequence ngrams and ids')
@ck.option(
    '--model-file', '-mf', default='data/model-interactions.h5',
    help='Model trained to predict interactions from sequences')
@ck.option(
    '--logger-file', '-mf', default='data/interactions-training.csv',
    help='Training logs for interactions')
@ck.option(
    '--load', '-ld', is_flag=True,
    help='Load model?')
@ck.option(
    '--batch-size', '-bs', default=512,
    help='Train batch size')
@ck.option(
    '--epochs', '-es', default=128,
    help='Train epochs')
def main(data_file, proteins_file, model_file, logger_file, load,
         batch_size, epochs):
    train_df, valid_df, test_df = load_data(data_file)
    prot_df = pd.read_pickle(proteins_file)

    ngrams = {}
    
    for i, row in enumerate(prot_df.itertuples()):
        ngrams[row.mappings] = row.sequences
        
    if load:
        logging.info('Loading pretrained model')
        model = load_model(model_file)
    else:
        logging.info('Creating a new model')
        model = create_model()

        logging.info(f"Training data size: {len(train_df)}")
        logging.info(f"Validation data size: {len(valid_df)}")
        checkpointer = ModelCheckpoint(
            filepath=model_file,
            verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=16, verbose=1)
        logger = CSVLogger(logger_file)

        logging.info('Starting training the model')

        valid_steps = int(math.ceil(len(valid_df) / batch_size))
        train_steps = int(math.ceil(len(train_df) / batch_size))
        train_generator = DFGenerator(train_df, ngrams, batch_size)
        valid_generator = DFGenerator(valid_df, ngrams, batch_size)
    
        model.summary()
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            max_queue_size=batch_size,
            workers=12,
            callbacks=[logger, checkpointer, earlystopper])
        logging.info('Loading best model')
        model = load_model(model_file)

    
    logging.info('Evaluating model')
    test_steps = int(math.ceil(len(test_df) / batch_size))
        
    test_generator = DFGenerator(test_df, ngrams, batch_size)

    loss = model.evaluate_generator(test_generator, steps=test_steps)
    logging.info(f'Test loss {loss}')
    
    
def create_model():
    inp1 = Input(shape=(MAXLEN,))
    inp2 = Input(shape=(MAXLEN,))
    
    embedding_dims = 10 
    max_features = 8001
    embedding_layer = Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN)
    conv_layer = Conv1D(
        filters=8,
        kernel_size=11,
        padding='valid',
        activation='relu',
        strides=1)
    pool_layer = MaxPooling1D(pool_size=3)
    dense_layer = Dense(256, activation='relu')
    flatten = Flatten()
    
    def get_features(inp):
        net = embedding_layer(inp)
        net = conv_layer(net)
        net = pool_layer(net)
        net = flatten(net)
        net = dense_layer(net)
        return net
    
    net1 = get_features(inp1)
    net2 = get_features(inp2)
    net = Dot(axes=1)([net1, net2])
    output = Activation('sigmoid')(net)

    model = Model(inputs=[inp1, inp2], outputs=output)
    model.summary()
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy')
    logging.info('Compilation finished')

    return model



def load_data(data_file, split=0.8):
    df = pd.read_pickle(data_file)
    n = 10000000 #len(df)

    # Split train/valid/test
    index = np.arange(n)
    valid_n = int(n * split)
    train_n = int(valid_n * split)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n: valid_n]]
    test_df = df.iloc[index[valid_n:]]

    return train_df, valid_df, test_df


class DFGenerator(object):

    def __init__(self, df, ngrams, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.ngrams = ngrams
        self.proteins = np.array(list(ngrams.keys()))
        
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            positives = []
            negatives = []
            for i, row in enumerate(df.itertuples()):
                positives.append(row.data)
                p = np.random.choice(self.proteins, 2)
                pair = (p[0], p[1])
                negatives.append(pair)
                    
            pairs = positives + negatives
            labels = np.array([1] * len(positives) + [0] * len(negatives))
            index = np.arange(len(pairs))
            np.random.shuffle(index)
            labels = labels[index].reshape(-1, 1)

            data_seq1 = np.zeros((len(df) * 2, MAXLEN), dtype=np.int32)
            data_seq2 = np.zeros((len(df) * 2, MAXLEN), dtype=np.int32)
            
            for i, j in enumerate(index):
                p1, p2 = pairs[j]
                seq1 = self.ngrams[p1]
                seq2 = self.ngrams[p2]
                data_seq1[i, 0:len(seq1)] = seq1
                data_seq2[i, 0:len(seq2)] = seq2

            self.start += self.batch_size
            data = [data_seq1, data_seq2]
            return (data, labels)
        else:
            self.reset()
            return self.next()
    

                


if __name__ == '__main__':
    main()
