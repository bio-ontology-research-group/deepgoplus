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
    '--cl-file', '-cf', default='data/cl-basic.obo',
    help='Cell Ontology file in OBO Format')
@ck.option(
    '--uberon-file', '-uf', default='data/uberon-basic.obo',
    help='Uberon Ontology file in OBO Format')
@ck.option(
    '--old-data-file', '-oldf', default='data/swissprot_exp201601.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--new-data-file', '-ndf', default='data/swissprot_exp201610.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--mapping-file', '-tf', default='data/go_mappings.txt',
    help='A mapping file extracted from definitions (mappings.py)')
@ck.option(
    '--embed-mapping-file', '-emf', default='data/diamond_mapping.out',
    help='Diamond mapping to stringdb proteins')
@ck.option(
    '--embed-file', '-ef', default='data/string/embeddings.pkl',
    help='Pandas DF with embeddings')
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
def main(go_file, cl_file, uberon_file, old_data_file, new_data_file,
         mapping_file, embed_mapping_file, embed_file,
         out_terms_file, train_data_file, test_data_file, min_count):
    go = Ontology(go_file, with_rels=True)
    logging.info('GO loaded')
    # cl = Ontology(cl_file)
    # logging.info('CL loaded')
    # uber = Ontology(uberon_file)
    # logging.info('UBERON loaded')
    
    df = pd.read_pickle(old_data_file)
    
    logging.info('Processing annotations')
    # Read mapping
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            it = list(map(lambda x: x.replace('_', ':'), it))
            mapping[it[0]] = it[1:]

    cnt = Counter()
    annotations = list()
    for i, row in df.iterrows():
        annots = set()
        for go_id in row['annotations']:
            if go_id in mapping:
                annots |= set(mapping[go_id])
            elif go.has_term(go_id):
                annots.add(go_id)
            
        annotations.append(annots)
        for term in row['annotations']:
            cnt[term] += 1
        # logging.info(f'Protein {row.proteins} processed')
        

    # Add mapped terms to data and save
    df['pred_annotations'] = annotations

    logging.info('Processing embeddings')
    # Add embedding features
    embed_dict = {}
    embed_map = {}
    with open(embed_mapping_file) as f:
        for line in f:
            it = line.strip().split()
            embed_map[it[1]] = it[0]
    embed_df = pd.read_pickle(embed_file)
    embed_size = 0
    for row in embed_df.itertuples():
        p_id = row.proteins
        if p_id in embed_map:
            embed_dict[embed_map[p_id]] = row.embeddings
            embed_size = len(row.embeddings)
    logging.info(f'Number of embeddings loaded {len(embed_dict)}')
    logging.info(f'Embedding size {embed_size}')
    embeddings = []
    train_prots = set()
    for row in df.itertuples():
        p_id = row.proteins
        train_prots.add(p_id)
        if p_id in embed_dict:
            embeddings.append(embed_dict[p_id])
        else:
            embeddings.append(np.zeros(embed_size, dtype='float32'))

    df['embeddings'] = embeddings
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

    embeddings = []
    index = []
    for i, row in enumerate(df.itertuples()):
        p_id = row.proteins
        if p_id not in train_prots:
            index.append(i)
            if p_id in embed_dict:
                embeddings.append(embed_dict[p_id])
            else:
                embeddings.append(np.zeros(embed_size, dtype='float32'))
    df = df.iloc[index]
    df['embeddings'] = embeddings

    annotations = list()
    for i, row in df.iterrows():
        annots = set()
        for go_id in row['annotations']:
            if go_id in mapping:
                annots |= set(mapping[go_id])
            elif go.has_term(go_id):
                annots.add(go_id)
            
        annotations.append(annots)
    df['pred_annotations'] = annotations
    
    df.to_pickle(test_data_file)

                


if __name__ == '__main__':
    main()
