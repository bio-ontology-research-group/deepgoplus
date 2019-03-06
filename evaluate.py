#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
from utils import Ontology, FUNC_DICT

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--ont', '-o', default='bp',
    help='GO subontology (mf, bp, cc)')
@ck.option(
    '--pred-file', '-pf', default='data/predictions.pkl',
    help='predictions file')
@ck.option(
    '--mapping-file', '-mpf', default='data/go_mappings.txt',
    help='Definition mapping file')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
def main(go_file, ont, pred_file, mapping_file, terms_file, threshold):
    go = Ontology(go_file, with_rels=False)
    go_rels = Ontology(go_file, with_rels=True)

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    go_set = go.get_term_set(FUNC_DICT[ont])        
    deepgo_funcs = pd.read_pickle('data/deepgo/' + ont + '.pkl')['functions'].values
    # go_set = set(deepgo_funcs.flatten())
    print(len(go_set))
    df = pd.read_pickle(pred_file)

    # Extend predictions using mapping
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            it = list(map(lambda x: x.replace('_', ':'), it))
            mapping[it[0]] = it[1:]

    logging.info('Number of definitions: %d' % len(mapping))
    predicted_annotations = list()
    preds = df['preds'].values
    for i in range(preds.shape[0]):
        annots = set()
        for j in range(len(preds[i])):
            if preds[i][j] >= threshold and terms[j].startswith('GO'):
                annots.add(terms[j])
        for t_id, items in mapping.items():
            ok = True
            for it in items:
                if it not in terms_dict or preds[i][terms_dict[it]] < threshold:
                    ok = False
                    break
            if ok:
                annots.add(t_id)
        # propagate annotations
        new_annots = set()
        for go_id in annots:
            new_annots |= go_rels.get_anchestors(go_id)
        predicted_annotations.append(annots)

    f_annots = compute_fscore_annotations(df['annotations'].values, predicted_annotations)
    logging.info('F score after expanding: %.2f' % f_annots)
    
    # Filter classes
    annots = list(map(lambda x: set(filter(lambda y: y in go_set, x)), df['annotations']))
    annots_set = set()
    for ann in annots:
        annots_set |= ann
    preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), predicted_annotations))
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
