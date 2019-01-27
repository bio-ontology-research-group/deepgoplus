#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import math

from utils import GeneOntology, read_fasta
from aminoacids import MAXLEN, to_ngrams

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, MaxPooling1D, Flatten, Dot, Activation
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K

from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)

config = tf.ConfigProto(allow_soft_placement=True)
session = tf.Session(config=config)
K.set_session(session)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/string/embeddings.pkl',
    help='String protein sequences and embeddings file')
@ck.option(
    '--model-file', '-mf', default='data/model-embeddings.h5',
    help='Model trained to predict embedding vector from sequences')
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
@ck.option(
    '--device', '-d', default='gpu:0',
    help='Device')
def main(data_file,  model_file, logger_file, load,
         batch_size, epochs, device):
    train_df, valid_df, test_df = load_data(data_file)

    with tf.device('/' + device):
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
            train_generator = DFGenerator(train_df, batch_size)
            valid_generator = DFGenerator(valid_df, batch_size)

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

        test_generator = DFGenerator(test_df, batch_size)

        loss = model.evaluate_generator(test_generator, steps=test_steps)
        logging.info(f'Test loss {loss}')

        test_generator.reset()
        preds = model.predict_generator(test_generator, steps=test_steps)

    for i, row in enumerate(test_df.itertuples()):
        X = row.embeddings
        X = X.reshape(1, 256)
        Y = preds[i].reshape(1, 256)
        sim = cosine_similarity(X, Y)
        print(sim)
        if i > 100:
            break
    
def create_model():
    inp = Input(shape=(MAXLEN,))
    
    embedding_dims = 128
    max_features = 8001
    embedding_layer = Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN)
    conv_layer = Conv1D(
        filters=32,
        kernel_size=16,
        padding='valid',
        activation='relu',
        strides=1)
    pool_layer = MaxPooling1D(pool_size=3)
    dense_layer = Dense(256)
    flatten = Flatten()

    net = embedding_layer(inp)
    net = conv_layer(net)
    net = pool_layer(net)
    net = flatten(net)
    net = dense_layer(net)
    
    model = Model(inputs=inp, outputs=net)
    model.summary()
    model.compile(
        optimizer='adam',
        loss='mae')
    logging.info('Compilation finished')

    return model



def load_data(data_file, split=0.8):
    df = pd.read_pickle(data_file)
    n = len(df)

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
            data_embed = np.zeros((len(df), 256), dtype=np.int32)
            data_seq = np.zeros((len(df), MAXLEN), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                seq = row.sequences
                data_seq[i, 0:len(seq)] = seq
                data_embed[i, :] = row.embeddings
                self.start += self.batch_size
            return (data_seq, data_embed)
        else:
            self.reset()
            return self.next()
    

                


if __name__ == '__main__':
    main()
