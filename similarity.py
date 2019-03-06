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
from sklearn.metrics.pairwise import cosine_similarity
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
from utils import FUNC_DICT, Ontology

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--train-data-file', '-trdf', default='data/train_data.pkl',
    help='Data file with training features')
@ck.option(
    '--test-data-file', '-tsdf', default='data/predictions.pkl',
    help='Test data file')
@ck.option(
    '--ont', '-o', default='bp',
    help='GO subontology (bp, mf, cc)')
def main(train_data_file, test_data_file, ont):
    go = Ontology('data/go.obo', with_rels=False)
    go_rels = Ontology('data/go.obo', with_rels=True)
    terms_df = pd.read_pickle('data/terms.pkl')
    terms = terms_df['terms'].values.flatten()
    
    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    annotations = train_df['annotations'].values
    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    # n_embeds = len(test_df['embeddings'].values[0])
    # train_embeds = np.zeros((len(train_df), n_embeds), dtype=np.float32)
    # for i, row in enumerate(train_df.itertuples()):
    #     train_embeds[i, :] = row.embeddings
    # test_embeds = np.zeros((len(test_df), n_embeds), dtype=np.float32)
    # for i, row in enumerate(test_df.itertuples()):
    #     test_embeds[i, :] = row.embeddings
    # sim1 = cosine_similarity(test_embeds, train_embeds)

    # n_embeds = len(test_df['features'].values[0])
    # train_embeds = np.zeros((len(train_df), n_embeds), dtype=np.float32)
    # for i, row in enumerate(train_df.itertuples()):
    #     train_embeds[i, :] = row.features
    # test_embeds = np.zeros((len(test_df), n_embeds), dtype=np.float32)
    # for i, row in enumerate(test_df.itertuples()):
    #     test_embeds[i, :] = row.features
    # sim2 = cosine_similarity(test_embeds, train_embeds)

    # BLAST Similarity (Diamond)
    pred_map = {}
    with open('data/test_data.res') as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in pred_map:
                pred_map[it[0]] = []
            pred_map[it[0]].append(it[1])

    preds = []
    top = 1
    for i, row in enumerate(test_df.itertuples()):
        # index1 = np.argsort(-sim1[i, :])
        # index2 = np.argsort(-sim2[i, :])
        annots = set()
        # DeepGOPLUS predictions
        for j, score in enumerate(row.preds):
            if score >= 0.2 and terms[j].startswith('GO:'):
                annots.add(terms[j])
            
        # for j in range(top):
        #     annots |= set(annotations[index1[j]])
        # for j in range(top):
        #     annots |= set(annotations[index2[j]])
        
        if row.proteins in pred_map:
            for prot_id in pred_map[row.proteins]:
                annots |= set(annotations[prot_index[prot_id]])

        new_annots = set()
        for go_id in annots:
            new_annots |= go_rels.get_anchestors(go_id)
        preds.append(new_annots)
        
    labels = test_df['annotations'].values

    go_set = go.get_term_set(FUNC_DICT[ont])        
    # deepgo_funcs = pd.read_pickle('data/deepgo/' + ont + '.pkl')['functions'].values
    # go_set = set(deepgo_funcs.flatten())

    print(len(go_set))
    
    # Filter classes
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
    
    print(compute_fscore_annotations(labels, preds))
        

def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

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


if __name__ == '__main__':
    main()
