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
    '--data-file', '-ndf', default='data/swissprot.pkl',
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
def main(go_file, data_file,
         out_terms_file, train_data_file, test_data_file, min_count):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')

    df = pd.read_pickle(data_file)
    print("DATA FILE" ,len(df))
    
    logging.info('Processing annotations')
    
    cnt = Counter()
    annotations = list()
    for i, row in df.iterrows():
        for term in row['prop_annotations']:
            cnt[term] += 1
    
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
    terms_df = pd.DataFrame({'terms': terms})
    terms_df.to_pickle(out_terms_file)

    n = len(df)
    # Split train/valid
    index = np.arange(n)
    train_n = int(n * 0.9)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = df.iloc[index[:train_n]]
    test_df = df.iloc[index[train_n:]]

    print('Number of train proteins', len(train_df))
    train_df.to_pickle(train_data_file)

    print('Number of test proteins', len(test_df))
    test_df.to_pickle(test_data_file)


if __name__ == '__main__':
    main()
