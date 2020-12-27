#!/usr/bin/env python

import numpy as np
import pandas as pd
import click as ck
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from scipy.spatial import distance
from scipy import sparse
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
from evaluate_deepgoplus import compute_mcc, compute_roc, evaluate_annotations
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--train-data-file', '-trdf', default='data/train_data.pkl',
    help='Data file with training features')
@ck.option(
    '--test-data-file', '-tsdf', default='data/predictions.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--diamond-scores-file', '-dsf', default='data/test_diamond.res',
    help='Diamond output')
@ck.option(
    '--ont', '-o', default='mf',
    help='GO subontology (bp, mf, cc)')


def main(train_data_file, test_data_file, terms_file,
         diamond_scores_file, ont):

    go_rels = Ontology('data/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    print("Length of test set: " + str(len(test_df)))
    
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    
    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])

    blast_preds = []
    #print('Diamond preds')
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= annotations[prot_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[prot_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            ind = np.argsort(-sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)
        
 

    find_alpha(ont, test_df, blast_preds, go_rels, terms)





################ GENETIC ALGORITHM TO FIND BEST ALPHAS PARAMETER ##################################

def eval_alphas(alpha, ont, test_df, blast_preds, go_rels, terms):


    deep_preds = []
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    
    labels = test_df['prop_annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

    alphas = {NAMESPACES['mf']: 0, NAMESPACES['bp']: 0, NAMESPACES['cc']: 0}
    alphas[NAMESPACES[ont]] = alpha    

    for i, row in enumerate(test_df.itertuples()):
        annots_dict = blast_preds[i].copy()
        for go_id in annots_dict:
            annots_dict[go_id] *= alphas[go_rels.get_namespace(go_id)]
        for j, score in enumerate(row.preds):
            go_id = terms[j]
            score *= 1 - alphas[go_rels.get_namespace(go_id)]
            if go_id in annots_dict:
                annots_dict[go_id] += score
            else:
                annots_dict[go_id] = score
        deep_preds.append(annots_dict)

    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    for t in range(1, 101): # the range in this loop has influence in the AUPR output
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for go_id, score in deep_preds[i].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)
            
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
    
        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, labels, preds)
        avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
        avg_ic = sum(map(lambda x: sum(map(lambda go_id: go_rels.get_ic(go_id), x)), fps)) / len(fps)
        #print(f'{avg_fp} {avg_ic}')
        precisions.append(prec)
        recalls.append(rec)
        # print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)

    if ont == 'mf':
        smin /= 15
    elif ont == 'bp':
        smin /= 40
    elif ont == 'cc':
        smin /= 10

    return alpha, np.sum([smin, -fmax, -aupr])




def find_alpha(ont, test_df, blast_preds, go_rels, terms):

    extra = [ont, test_df, blast_preds, go_rels, terms]
    inputs = range(45, 75, 1)

    num_cores = 30

    results = Parallel(n_jobs=num_cores)(delayed(eval_alphas)(i/100, *extra) for i in inputs)

    print(results)




#####################################################################################

if __name__ == '__main__':
    main()
