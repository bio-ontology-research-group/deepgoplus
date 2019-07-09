#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from subprocess import Popen, PIPE
import time
from utils import Ontology, NAMESPACES, FUNC_DICT
from aminoacids import to_onehot
import math
from collections import Counter

MAXLEN = 2000

@ck.command()
@ck.option('--model-file', '-mf', default='data-cafa/model.h5', help='Tensorflow model file')
@ck.option('--terms-file', '-tf', default='data-cafa/terms.pkl', help='List of predicted terms')
@ck.option('--annotations-file', '-tf', default='data-cafa/swissprot.pkl', help='Experimental annotations')
def main(model_file, terms_file, annotations_file):

    go_rels = Ontology('data-cafa/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: k for k, v in  enumerate(terms)}
    df = pd.read_pickle(annotations_file)
    annotations = df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    go_rels.calculate_ic(annotations)

    go_id = 'GO:0008047'
    go_idx = terms_dict[go_id]
    # df = df[df['orgs'] == '559292']

    index = []
    seq_lengths = []
    for i, row in enumerate(df.itertuples()):
        if go_id in row.annotations:
            index.append(i)
            seq_lengths.append(len(row.sequences))
    df = df.iloc[index]
    
    annotations = df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    
    
    prot_ids = df['proteins'].values
    ids, data = get_data(df['sequences'])
    # for i, row in df.iterrows():
    #     ipros = '\t'.join(row['interpros'])
    #     print(f'{row["proteins"]}\t{ipros}')
    # Load CNN model
    model = load_model(model_file)
    model.summary()
    return
    int_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    dense = model.layers[-1]
    W = dense.get_weights()[0][:, go_idx]
    b = dense.get_weights()[1][go_idx]
    print(np.argsort(-W), b)
    preds = int_model.predict(data, batch_size=100, verbose=0)
    filters = np.argsort(preds, axis=1)
    filter_cnt = Counter()
    for f in filters:
        filter_cnt.update(f[:20])
    AALETTER = np.array([
        '*', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    print(filter_cnt)
    return
    for f_id, cnt in filter_cnt.most_common(10):
        conv_id = f_id // 512
        fl_id = f_id % 512
        conv_layer = model.layers[conv_id + 1]
        weights = conv_layer.get_weights()
        w1 = weights[0]
        w2 = weights[1]
        motif = ''.join(AALETTER[np.argmax(w1[:, :, fl_id], axis=1)])
        print(f'>{f_id}')
        print(motif)
        conv_model = Model(inputs=model.input, outputs=conv_layer.output)
        preds = conv_model.predict(data, batch_size=100, verbose=0)
        f_out = preds[:, :, fl_id]
        f_length = conv_layer.kernel_size[0]
        starts = np.argmax(f_out, axis=1)
        ends = starts + f_length
        for i in range(starts.shape[0]):
            seq = data[i, starts[i]:ends[i], :]
            seq_ind = np.argmax(seq, axis=1)
            motif = ''.join(AALETTER[seq_ind])
            print(f'>{f_id}_{i}')
            print(motif.replace('*', ''))
    # for l in range(16):
    #     conv1 = model.layers[l + 1]
    #     weights = conv1.get_weights()
    #     w1 = weights[0]
    #     w2 = weights[1]
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
