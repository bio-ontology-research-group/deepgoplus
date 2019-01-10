#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd

from utils import GeneOntology, FUNC_DICT

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--ont', '-o', default='bp',
    help='GO subontology (mf, bp, cc)')
@ck.option(
    '--pred-file', '-pf', default='data/predictions.pkl',
    help='GO subontology (mf, bp, cc)')
def main(go_file, ont, pred_file):
    go = GeneOntology(go_file, with_rels=False)

    # go_set = go.get_go_set(FUNC_DICT[ont])        
    deepgo_funcs = pd.read_pickle('data/deepgo/' + ont + '.pkl')['functions'].values
    go_set = set(deepgo_funcs.flatten())
    print(len(go_set))
    df = pd.read_pickle(pred_file)

    # Filter classes
    annots = list(map(lambda x: set(filter(lambda y: y in go_set, x)), df['annotations']))
    annots_set = set()
    for ann in annots:
        annots_set |= ann
    preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), df['predicted_annotations']))
    preds_set = set()
    for ann in preds:
        preds_set |= ann
    print(len(annots_set), len(preds_set), len(annots_set.intersection(preds_set)))
    f = compute_fscore_annotations(annots, preds)
    print('F score: %.2f' % f)

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
