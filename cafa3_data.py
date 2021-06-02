#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, FUNC_DICT, read_fasta
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data-cafa3/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--train-sequences-file', '-trsf', default='data-cafa3/CAFA3_training_data/uniprot_sprot_exp.fasta',
    help='CAFA training sequences fasta')
@ck.option(
    '--train-annotations-file', '-traf', default='data-cafa3/CAFA3_training_data/uniprot_sprot_exp.txt',
    help='CAFA training annotations fasta')
@ck.option(
    '--test-sequences-file', '-tssf', default='data-cafa3/CAFA3_targets/targets_all.fasta',
    help='CAFA training annotations fasta')
@ck.option(
    '--test-annotations-file', '-tsaf', default='data-cafa3/benchmark20171115/groundtruth/leafonly_all.txt',
    help='CAFA training annotations fasta')
@ck.option(
    '--out-terms-file', '-otf', default='data-cafa3/terms.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--train-data-file', '-trdf', default='data-cafa3/train_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--test-data-file', '-tsdf', default='data-cafa3/test_data.pkl',
    help='Result file with a list of terms for prediction task')
@ck.option(
    '--min-count', '-mc', default=50,
    help='Minimum number of annotated proteins')
def main(go_file, train_sequences_file, train_annotations_file,
         test_sequences_file, test_annotations_file,
         out_terms_file, train_data_file, test_data_file, min_count):
    logging.info('Loading GO')
    go = Ontology(go_file, with_rels=True)
    
    logging.info('Loading training annotations')
    train_annots = {}
    with open(train_annotations_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            if prot_id not in train_annots:
                train_annots[prot_id] = set()
            go_id = it[1]
            train_annots[prot_id].add(go_id)
                
    logging.info('Loading training sequences')
    info, seqs = read_fasta(train_sequences_file)
    proteins = []
    sequences = []
    annotations = []
    for prot_info, sequence in zip(info, seqs):
        prot_id = prot_info.split()[0]
        if prot_id in train_annots:
            proteins.append(prot_id)
            sequences.append(sequence)
            annotations.append(train_annots[prot_id])
    
    prop_annotations = []
    cnt = Counter()
    for annots in annotations:
        # Propagate annotations
        annots_set = set()
        for go_id in annots:
            annots_set |= go.get_anchestors(go_id)
        prop_annotations.append(annots_set)
        for go_id in annots_set:
            cnt[go_id] += 1
        
    df = pd.DataFrame({
        'proteins': proteins,
        'sequences': sequences,
        'annotations': prop_annotations,
    })
    logging.info(f'Train proteins: {len(df)}')
    logging.info(f'Saving training data to {train_data_file}')
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
        terms += val

    logging.info(f'Number of terms {len(terms)}')
    logging.info(f'Saving terms to {out_terms_file}')
    
    df = pd.DataFrame({'terms': terms})
    df.to_pickle(out_terms_file)

    logging.info('Loading testing annotations')
    test_annots = {}
    with open(test_annotations_file, 'r') as f:
        for line in f:
            it = line.strip().split('\t')
            prot_id = it[0]
            if prot_id not in test_annots:
                test_annots[prot_id] = set()
            go_id = it[1]
            test_annots[prot_id].add(go_id)
                
    logging.info('Loading testing sequences')
    info, seqs = read_fasta(test_sequences_file)
    proteins = []
    sequences = []
    annotations = []
    for prot_info, sequence in zip(info, seqs):
        prot_id = prot_info.split()[0]
        if prot_id in test_annots:
            proteins.append(prot_id)
            sequences.append(sequence)
            annotations.append(test_annots[prot_id])
    
    prop_annotations = []
    for annots in annotations:
        # Propagate annotations
        annots_set = set()
        for go_id in annots:
            annots_set |= go.get_anchestors(go_id)
        prop_annotations.append(annots_set)
        
    df = pd.DataFrame({
        'proteins': proteins,
        'sequences': sequences,
        'annotations': prop_annotations,
    })
    logging.info(f'Test proteins {len(df)}')
    logging.info(f'Saving testing data to {test_data_file}')
    df.to_pickle(test_data_file)

                


if __name__ == '__main__':
    main()
