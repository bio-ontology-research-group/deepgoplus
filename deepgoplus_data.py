#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--old-data-file', '-oldf', default='data/swissprot_exp_2019_03.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--new-data-file', '-ndf', default='data/swissprot_exp_2019_10.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--out-terms-file', '-otf', default='data/terms.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--train-data-file', '-trdf', default='data/train_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--test-data-file', '-tsdf', default='data/test_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--min-count', '-mc', default=50,
    help='Minimum number of annotated proteins')
def main(go_file, old_data_file, new_data_file,
         out_terms_file, train_data_file, test_data_file, min_count):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    df = pd.read_pickle(old_data_file)
    
    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    for i, row in df.iterrows():
        for term in row['annotations']:
            cnt[term] += 1
    
    train_prots = set()
    for row in df.itertuples():
        p_id = row.proteins
        train_prots.add(p_id)

    df.to_pickle(train_data_file)

    # Filter terms with annotations more than min_count
    res = {}
    for key, val in cnt.items():
        if val >= min_count:
            ont = key.split(':')[0]
            if ont not in res:
                res[ont] = []
            res[ont].append(key)
    terms = []
    for key, val in res.items():
        print(key, len(val))
        terms += val

    logging.info(f'Number of terms {len(terms)}')
    
    # Save the list of terms
    df = pd.DataFrame({'terms': terms})
    df.to_pickle(out_terms_file)

    # Save testing data
    df = pd.read_pickle(new_data_file)

    index = []
    for i, row in enumerate(df.itertuples()):
        p_id = row.proteins
        if p_id not in train_prots:
            index.append(i)
    df = df.iloc[index]
    print('Number of test proteins', len(df))
    df.to_pickle(test_data_file)
                


if __name__ == '__main__':
    main()
