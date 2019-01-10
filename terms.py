#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import GeneOntology

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='Uniprot KB, generated with uni2pandas.py')
@ck.option(
    '--mapping-file', '-tf', default='data/mappings.txt',
    help='A mapping file extracted from definitions (mappings.py)')
@ck.option(
    '--out-terms-file', '-otf', default='data/terms.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--out-data-file', '-odf', default='data/data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--min-count', '-mc', default=50,
    help='Minimum number of annotated proteins')
def main(go_file, data_file, mapping_file, out_terms_file, out_data_file, min_count):
    go = GeneOntology(go_file, with_rels=True)
    df = pd.read_pickle(data_file)

    # Read mapping
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            it = list(map(lambda x: x.replace('_', ':'), it))
            mapping[it[0]] = it[1:]

    # Get terms without definitions
    terms = set()
    for go_id, maps in mapping.items():
        for term in maps:
            if term not in mapping:
                terms.add(term)

    # Count annotations
    cnt = Counter()
    annotations = list()
    for i, row in df.iterrows():
        annots = set()
        for go_id in row['annotations']:
            if go_id in mapping:
                for term in mapping[go_id]:
                    if term not in mapping:
                        annots.add(term)
            else:
                annots.add(go_id)
        annotations.append(annots)
        for term in annots:
            cnt[term] += 1

    # Add mapped terms to data and save
    df['pred_annotations'] = annotations
    df.to_pickle(out_data_file)

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

    # Save the list of terms
    df = pd.DataFrame({'terms': terms})
    df.to_pickle(out_terms_file)

                


if __name__ == '__main__':
    main()
