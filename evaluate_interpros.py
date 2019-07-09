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
def main():
    pfam2ipro = {}
    with open('data/pfam2ipro.txt') as f:
        for line in f:
            it = line.strip().split()
            pfam2ipro[it[0]] = it[1]

    id2prot = open('data/seq_prot_ids.txt').read().splitlines()
    aligns = {}
    with open('data/test.align') as f:
        for line in f:
            it = line.strip().split()
            prot_idx = int(it[0].split('_')[1])
            prot_id = id2prot[prot_idx]
            pfam = it[1][:-1]
            ipro = pfam2ipro[pfam]
            if prot_id not in aligns:
                aligns[prot_id] = set()
            aligns[prot_id].add(ipro)
    interpros = {}
    with open('data/prot_interpros.txt') as f:
        for line in f:
            it = line.strip().split('\t')
            interpros[it[0]] = set(it[1:])
    preds = list()
    annots = list()
    for prot_id, ipros in aligns.items():
        preds.append(ipros)
        annots.append(interpros[prot_id])
    print(len(aligns))
    f, p, r = evaluate_annotations(annots, preds)
    print(f, p, r)

def evaluate_annotations(real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
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
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    return f, p, r

if __name__ == '__main__':
    main()
