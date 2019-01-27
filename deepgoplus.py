#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv2D, Flatten, Concatenate,
    MaxPooling2D, Dropout,
)
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from utils import Ontology, FUNC_DICT
from aminoacids import to_ngrams, to_onehot

MAXLEN = 1000
logging.basicConfig(level=logging.INFO)

config = tf.ConfigProto(allow_soft_placement=True)
session = tf.Session(config=config)
K.set_session(session)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--train-data-file', '-df', default='data/train_data.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--test-data-file', '-df', default='data/test_data.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--interpros-file', '-ipf', default='data/interpros.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--mapping-file', '-mpf', default='data/mappings.txt',
    help='Definition mapping file')
@ck.option(
    '--model-file', '-mf', default='data/model.h5',
    help='DeepGOPlus model')
@ck.option(
    '--interactions-model-file', '-imf', default='data/model-interactions.h5',
    help='Model for predicting interactions')
@ck.option(
    '--interpro-model-file', '-ipmf', default='data/model_ipro.h5',
    help='Model for predicting InterPro domains')
@ck.option(
    '--sequence-model-file', '-smf', default='data/model_seq.h5',
    help='Sequence2sequence autoencoder model')
@ck.option(
    '--out-file', '-o', default='data/predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--split', '-s', default=0.9,
    help='train/test split')
@ck.option(
    '--batch-size', '-bs', default=128,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=128,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data/training.csv',
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
@ck.option(
    '--device', '-d', default='gpu:0',
    help='Prediction threshold')
@ck.option(
    '--term-index', '-ti', default=-1,
    help='Index of term for binary prediction')
def main(go_file, train_data_file, test_data_file, terms_file, interpros_file, mapping_file,
         model_file, interactions_model_file, interpro_model_file,
         sequence_model_file, out_file,
         split, batch_size, epochs, load, logger_file, threshold,
         device, term_index):
    # model = create_model(interactions_model_file, 1000)
    # return
    go = Ontology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    balance = False
    if term_index != -1:
        terms = terms[term_index:term_index + 1]
        balance=True

    train_df, valid_df = load_data(
        train_data_file, terms, split, balance=balance)
    test_df = pd.read_pickle(test_data_file)
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)

    ipros_df = pd.read_pickle(interpros_file)
    interpros = ipros_df['interpros']
    ipros_dict = {v: i for i, v in enumerate(interpros)}
    nb_ipros = len(interpros)
    
    with tf.device('/' + device):
        if load:
            logging.info('Loading pretrained model')
            model = load_model(model_file)
        else:
            logging.info('Creating a new model')
            model = create_model(
                interactions_model_file, interpro_model_file,
                nb_classes, nb_ipros)
            
            logging.info("Training data size: %d" % len(train_df))
            logging.info("Validation data size: %d" % len(valid_df))
            checkpointer = ModelCheckpoint(
                filepath=model_file,
                verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=16, verbose=1)
            logger = CSVLogger(logger_file)

            logging.info('Starting training the model')

            valid_steps = int(math.ceil(len(valid_df) / batch_size))
            train_steps = int(math.ceil(len(train_df) / batch_size))
            train_generator = DFGenerator(train_df, terms_dict, ipros_dict,
                                          nb_classes, batch_size)
            valid_generator = DFGenerator(valid_df, terms_dict, ipros_dict,
                                          nb_classes, batch_size)
    
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
        
        test_generator = DFGenerator(test_df, terms_dict, ipros_dict,
                                     nb_classes, batch_size)
    
        loss = model.evaluate_generator(test_generator, steps=test_steps)
        logging.info('Test loss %f' % loss)
        
        logging.info('Predicting')
        test_generator.reset()
        preds = model.predict_generator(test_generator, steps=test_steps)
    
    test_labels = np.zeros((len(test_df), nb_classes), dtype=np.int32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.pred_annotations:
            if go_id in terms_dict:
                test_labels[i, terms_dict[go_id]] = 1
    logging.info('Computing performance:')
    roc_auc = compute_roc(test_labels, preds)
    logging.info('ROC AUC: %.2f' % (roc_auc,))
    predictions = (preds >= threshold).astype('int32')
    mcc = compute_mcc(test_labels, predictions)
    logging.info('MCC: %.2f' % (mcc,))
    f = compute_fscore(test_labels, predictions)
    logging.info('Fscore: %.2f' % (f,))
    test_df['labels'] = list(test_labels)
    test_df['preds'] = list(preds)
    
    logging.info('Saving predictions')
    test_df.to_pickle(out_file)

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, predictions):
    mcc = matthews_corrcoef(labels.flatten(), predictions.flatten())
    return mcc

def compute_fscore(labels, predictions):
    total = 0
    p_total = 0
    p = 0.0
    r = 0.0
    for i in range(labels.shape[0]):
        tp = np.sum(predictions[i, :] * labels[i, :])
        fp = np.sum(predictions[i, :]) - tp
        fn = np.sum(labels[i, :]) - tp
        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall
    r /= total
    p /= p_total
    if p + r > 0:
        return 2 * p * r / (p + r)
    return 0.0

def compute_fscore_annotations(real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = len(set(real_annots[i]).intersection(set(pred_annots[i])))
        fp = len(pred_annots[i]) - tp
        fn = len(real_annots[i]) - tp
        total += 1
        recall = tp / (1.0 * (tp + fn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            p += precision
            
    r /= total
    p /= p_total
    if p + r > 0:
        return 2 * p * r / (p + r)
    return 0.0


def get_interaction_features(model_file, inp, trainable=False):
    model = load_model(model_file)
    if not trainable:
        for i in range(2, 7):
            model.layers[i].trainable = trainable
    embed = model.layers[2]
    conv = model.layers[3]
    pool = model.layers[4]
    flatten = model.layers[5]
    dense = model.layers[6]
    net = dense(flatten(pool(conv(embed(inp)))))
    return net

def get_ipro_features(model_file, inp, trainable=False):
    model = load_model(model_file)
    if not trainable:
        for i in range(1, 6):
            model.layers[i].trainable = trainable
    embed = model.layers[1]
    conv = model.layers[2]
    pool = model.layers[3]
    flatten = model.layers[4]
    dense = model.layers[5]
    net = dense(flatten(pool(conv(embed(inp)))))
    return net


def create_model(int_model_file, ipro_model_file, nb_classes, nb_ipros):
    # inp = Input(shape=(MAXLEN,), dtype=np.int32)
    inp_hot = Input(shape=(MAXLEN, 21, 1), dtype=np.float32)
    inp_net = Input(shape=(256,), dtype=np.float32)
    inp_ipros = Input(shape=(nb_ipros,), dtype=np.float32)

    kernels = range(8, 129, 8)
    nets = []
    for i in range(len(kernels)):
        conv = Conv2D(
            filters=256,
            kernel_size=(kernels[i], 21),
            padding='valid',
            name='conv_1_' + str(i),
            activation='relu',
            kernel_initializer='glorot_normal')(inp_hot)
        # conv = Conv2D(
        #     filters=256,
        #     kernel_size=(kernels[i] * 2, 1),
        #     padding='valid',
        #     name='conv_2_' + str(i),
        #     activation='relu',
        #     kernel_initializer='glorot_normal')(conv)
        print(conv.get_shape())
        pool = MaxPooling2D(pool_size=(MAXLEN - kernels[i] + 1, 1))(conv)
        flat = Flatten()(pool)
        nets.append(flat)

    nets.append(inp_net)
    nets.append(inp_ipros)
    net = Concatenate(axis=1)(nets)
    net = Dense(
        nb_classes,
        activation='relu',
        name='dense1')(net)
    # net = Dropout(0.5)(net)
    output = Dense(nb_classes, activation='sigmoid', name='dense_out')(net)

    model = Model(inputs=[inp_hot, inp_net, inp_ipros], outputs=output)
    model.summary()
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy')
    logging.info('Compilation finished')

    return model



def load_data(data_file, terms, split, balance=False):
    df = pd.read_pickle(data_file)
    n = len(df)
    # Balance the dataset
    if balance:
        term = terms[0]
        pos = []
        neg = []
        for i, row in enumerate(df.itertuples()):
            if term in row.pred_annotations:
                pos.append(i)
            else:
                neg.append(i)
        np.random.shuffle(pos)
        np.random.shuffle(neg)
        n = min(len(pos), len(neg))
        pos = pos[:n]
        neg = neg[:n]
        pos_df = df.iloc[pos]
        neg_df = df.iloc[neg]
        df = pd.concat([pos_df, neg_df])
    # Split train/valid
    n = len(df)
    index = np.arange(n)
    train_n = int(n * split)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n:]]
    
    return train_df, valid_df
    

class DFGenerator(object):

    def __init__(self, df, terms_dict, ipros_dict, nb_classes, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.terms_dict = terms_dict
        self.ipros_dict = ipros_dict
        self.ipros_size = len(ipros_dict)
        
    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            df = self.df.iloc[batch_index]
            data_seq = np.zeros((len(df), MAXLEN), dtype=np.int32)
            data_onehot = np.zeros((len(df), MAXLEN, 21, 1), dtype=np.int32)
            data_net = np.zeros((len(df), 256), dtype=np.float32)
            data_ipros = np.zeros((len(df), self.ipros_size), dtype=np.float32)
            labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                seq = row.sequences[:(MAXLEN + 2)]
                ngrams = to_ngrams(seq)
                onehot = to_onehot(seq)
                data_seq[i, 0:len(ngrams)] = ngrams
                data_onehot[i, :, :, 0] = onehot
                data_net[i, :] = row.embeddings
                for i_id in row.interpros:
                    if i_id in self.ipros_dict:
                        data_ipros[i, self.ipros_dict[i_id]] = 1
                for t_id in row.pred_annotations:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return ([data_onehot, data_net, data_ipros], labels)
        else:
            self.reset()
            return self.next()

    
if __name__ == '__main__':
    main()
