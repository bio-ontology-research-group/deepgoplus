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
from matplotlib import pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@ck.command()
@ck.option(
    '--train-data-file', '-trdf', default='data-cafa3/train_data.pkl',
    help='Data file with training features')
@ck.option(
    '--test-data-file', '-tsdf', default='data-cafa3/predictions.pkl',
    help='Test data file')
@ck.option(
    '--terms-file', '-tf', default='data-cafa3/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--diamond-scores-file', '-dsf', default='data-cafa3/test_diamond.res',
    help='Diamond output')
@ck.option(
    '--ont', '-o', default='mf',
    help='GO subontology (bp, mf, cc)')
@ck.option(
    '--alpha', '-a', default=100.0,
    help='Alpha for for combining scores')
def main(train_data_file, test_data_file, terms_file,
         diamond_scores_file, ont, alpha):

    alpha /= 100.0
    go_rels = Ontology('data-cafa3/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)
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
    blast_tresh = {'mf': 0.12, 'cc': 0.27, 'bp':0.21 }
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
                # if score >= blast_tresh[ont]:
                annots[go_id] = score
        blast_preds.append(annots)
        
    # DeepGOPlus
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df['annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    print(len(go_set))
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    for t in range(1, 101):
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots_dict = blast_preds[i].copy()
            for go_id in annots_dict:
                annots_dict[go_id] *= alpha
            for j, score in enumerate(row.preds):
                go_id = terms[j]
                # Consensus between BLAST and DeepGOPLus
                # score = 1 - alpha * score
                # if go_id in blast_preds[i]:
                #     score *= (1 - alpha * blast_preds[i][go_id])
                # score = 1 - score
                score *= 1 - alpha
                if go_id in annots_dict:
                    annots_dict[go_id] += score
                else:
                    annots_dict[go_id] = score
                
            annots = set()
            for go_id, score in annots_dict.items():
                if score >= threshold:
                    annots.add(go_id)
            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)
        
    
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
    
        fscore, prec, rec, s = evaluate_annotations(go_rels, labels, preds)
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')
    plt.figure()
    lw = 2
    plt.plot(recalls, precisions, color='darkorange',
             lw=lw, label=f'AUPR curve (area = {aupr:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.savefig('aupr.pdf')
    plt.show()


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s


if __name__ == '__main__':
    main()
