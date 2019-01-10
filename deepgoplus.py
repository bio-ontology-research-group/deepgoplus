#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Conv1D, Flatten
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from utils import GeneOntology
from aminoacids import to_ngrams

MAXLEN = 1000
logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/data.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--mapping-file', '-mpf', default='data/mappings.txt',
    help='Definition mapping file')
@ck.option(
    '--model-file', '-mf', default='data/model.h5',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--out-file', '-o', default='data/predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--split', '-s', default=0.8,
    help='train/test split')
@ck.option(
    '--batch-size', '-bs', default=128,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=12,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data/training.csv',
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
def main(go_file, data_file, terms_file, mapping_file, model_file, out_file,
         split, batch_size, epochs, load, logger_file, threshold):
    go = GeneOntology(go_file, with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)
    train_df, valid_df, test_df = load_data(data_file, split)
    
    if load:
        logging.info('Loading pretrained model')
        model = load_model(model_file)
    else:
        logging.info('Creating a new model')
        model = create_model(nb_classes)

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
        train_generator = DFGenerator(train_df, terms_dict, nb_classes, batch_size)
        valid_generator = DFGenerator(valid_df, terms_dict, nb_classes, batch_size)
    
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
        
    test_generator = DFGenerator(test_df, terms_dict, nb_classes, batch_size)
    
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
    
    # Extend predictions using mapping
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            it = list(map(lambda x: x.replace('_', ':'), it))
            ok = True
            for t_id in it[1:]:
                if t_id not in terms_dict:
                    ok = False
                    break
            if ok:
                mapping[it[0]] = it[1:]
    logging.info('Number of definitions: %d' % len(mapping))
    predicted_annotations = list()
    for i in range(predictions.shape[0]):
        annots = set()
        for j in range(predictions.shape[1]):
            if predictions[i, j] == 1 and terms[j].startswith('GO'):
                annots.add(terms[j])
        for t_id, items in mapping.items():
            ok = True
            for it in items:
                if predictions[i, terms_dict[it]] == 0:
                    ok = False
                    break
            if ok:
                annots.add(t_id)
        # propagate annotations
        new_annots = set()
        for go_id in annots:
            new_annots |= go.get_anchestors(go_id)
        predicted_annotations.append(new_annots)

    f_annots = compute_fscore_annotations(test_df['annotations'].values, predicted_annotations)
    logging.info('F score after expanding: %.2f' % f_annots)
    
    test_df['predicted_annotations'] = predicted_annotations
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

        
def create_model(nb_classes):
    inp = Input(shape=(MAXLEN,))

    embedding_dims = 10 
    max_features = 8001
    embed = Embedding(
        max_features,
        embedding_dims,
        input_length=MAXLEN)(inp)
    net = Conv1D(
        filters=32,
        kernel_size=11,
        padding='valid',
        activation='relu',
        strides=1)(embed)
    net = Flatten()(net)
    output = Dense(nb_classes, activation='sigmoid')(net)

    model = Model(inputs=inp, outputs=output)
    model.summary()
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy')
    logging.info('Compilation finished')

    return model



def load_data(data_file, split):
    df = pd.read_pickle(data_file)
    n = len(df)

    # Split train/valid/test
    index = np.arange(len(df))
    valid_n = int(n * split)
    train_n = int(valid_n * split)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    valid_df = df.iloc[index[train_n: valid_n]]
    test_df = df.iloc[index[valid_n:]]

    return train_df, valid_df, test_df
    

class DFGenerator(object):

    def __init__(self, df, terms_dict, nb_classes, batch_size):
        self.start = 0
        self.size = len(df)
        self.df = df
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.terms_dict = terms_dict
        
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
            labels = np.zeros((len(df), self.nb_classes), dtype=np.int32)
            for i, row in enumerate(df.itertuples()):
                ngrams = to_ngrams(row.sequences[:(MAXLEN + 2)])
                data_seq[i, 0:len(ngrams)] = ngrams
                for t_id in row.pred_annotations:
                    if t_id in self.terms_dict:
                        labels[i, self.terms_dict[t_id]] = 1
            self.start += self.batch_size
            return (data_seq, labels)
        else:
            self.reset()
            return self.next()

    
if __name__ == '__main__':
    main()
