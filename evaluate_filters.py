#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import Popen, PIPE
import time
from utils import Ontology, NAMESPACES, FUNC_DICT
from aminoacids import to_onehot
import math

MAXLEN = 2000

@ck.command()
@ck.option('--model-file', '-mf', default='data-cafa/model.h5', help='Tensorflow model file')
@ck.option('--terms-file', '-tf', default='data-cafa/terms.pkl', help='List of predicted terms')
@ck.option('--annotations-file', '-tf', default='data-cafa/train_data.pkl', help='Experimental annotations')
def main(model_file, terms_file, annotations_file):

    go_rels = Ontology('data-cafa/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()

    df = pd.read_pickle(annotations_file)
    annotations = df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    go_rels.calculate_ic(annotations)
    
    # df = df[df['orgs'] == '559292']
    sl = 0

    annotations = df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    
    
    prot_ids = df['proteins'].values
    ids, data = get_data(df['sequences'])
    
    # Load CNN model
    model = load_model(model_file)

    preds = model.predict(data, batch_size=100, verbose=1)
    assert preds.shape[1] == len(terms)
    mf_set = go_rels.get_namespace_terms(NAMESPACES['mf'])
    # terms = ['GO:0008047']
    for l in range(len(terms)):
        # if terms[l] not in mf_set:
        #     continue
        deep_preds = {}
        for i, j in enumerate(ids):
            prot_id = prot_ids[j]
            if prot_id not in deep_preds:
                deep_preds[prot_id] = {}
            if preds[i, l] >= 0.01: # Filter out very low scores
                if terms[l] not in deep_preds[prot_id]:
                    deep_preds[prot_id][terms[l]] = preds[i, l]
                else:
                    deep_preds[prot_id][terms[l]] = max(
                        deep_preds[prot_id][terms[l]], preds[i, l])


        go_set = set([terms[l]])
        # go_set.remove(FUNC_DICT['mf'])
        labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), annotations))
        bin_labels = list(map(lambda x: len(x), labels))
        pos_cnt = sum(bin_labels)
        fmax = 0.0
        tmax = 0.0
        smin = 1000
        for t in range(0, 100):
            threshold = t / 100.0
            predictions = []
            for i, row in enumerate(df.itertuples()):
                annots_dict = deep_preds[row.proteins] or {}

                annots = set()
                for go_id, score in annots_dict.items():
                    if score >= threshold:
                        annots.add(go_id)
                # new_annots = set()
                # for go_id in annots:
                #     new_annots |= go_rels.get_anchestors(go_id)
                predictions.append(annots)


            # Filter classes
            predictions = list(map(lambda x: set(filter(lambda y: y in go_set, x)), predictions))

            fscore, prec, rec, s = evaluate_annotations(go_rels, labels, predictions)
            # print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
            if fmax < fscore:
                fmax = fscore
                tmax = threshold
            if smin > s:
                smin = s
        print(f'{terms[l]} {pos_cnt} Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
        # for l in range(16):
    #     conv1 = model.layers[l + 1]
    #     weights = conv1.get_weights()
    #     w1 = weights[0]
    #     w2 = weights[1]
    #     AALETTER = np.array([
    #         '*', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    #         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    #     for i in range(512):
    #         motif = ''.join(AALETTER[np.argmax(w1[:, :, i], axis=1)])
    #         print(f'>{l}_{i}')
    #         print(motif)



def get_data(sequences):
    pred_seqs = []
    ids = []
    for i, seq in enumerate(sequences):
        if len(seq) > MAXLEN:
            st = 0
            while st < len(seq):
                pred_seqs.append(seq[st: st + MAXLEN])
                ids.append(i)
                st += MAXLEN - 128
        else:
            pred_seqs.append(seq)
            ids.append(i)
    n = len(pred_seqs)
    data = np.zeros((n, MAXLEN, 21), dtype=np.float32)
    
    for i in range(n):
        seq = pred_seqs[i]
        data[i, :, :] = to_onehot(seq)
    return ids, data


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
    if total == 0:
        return 0, 0, 0, 1000
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s

if __name__ == '__main__':
    main()
