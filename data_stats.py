#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, NAMESPACES
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--terms-file', '-tf', default='data/terms.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--train-data-file', '-trdf', default='data/train_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--test-data-file', '-tsdf', default='data/test_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--ont', '-o', default='bp',
    help='Minimum number of annotated proteins')
def main(go_file, terms_file, train_data_file, test_data_file, ont):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    go_set = go.get_namespace_terms(NAMESPACES[ont])
    terms_df = pd.read_pickle(terms_file)
    tcnt = 0
    print('Total terms', len(terms_df))
    
    for go_id in terms_df['terms']:
        if go_id in go_set:
            tcnt += 1
    trdf = pd.read_pickle(train_data_file)            
    print('Total train', len(trdf))
    cnt = 0
    for i, row in trdf.iterrows():
        ok = False
        for go_id in row['annotations']:
            if go_id in go_set:
                ok = True
                break
        if ok:
            cnt += 1
    print('Number of training proteins', cnt)

    tsdf = pd.read_pickle(test_data_file)            
    print('Total test', len(tsdf))
    cnt = 0
    for i, row in tsdf.iterrows():
        ok = False
        for go_id in row['annotations']:
            if go_id in go_set:
                ok = True
                break
        if ok:
            cnt += 1
    print('Number of testing proteins', cnt)
    print('Number of terms', tcnt)

if __name__ == '__main__':
    main()
