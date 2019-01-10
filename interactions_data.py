#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip

from utils import GeneOntology, read_fasta
from aminoacids import MAXLEN, to_ngrams

@ck.command()
@ck.option(
    '--string-db-file', '-sdb', default='data/string/protein.links.v10.5.txt.gz',
    help='StringDB protein links file')
@ck.option(
    '--sequences-file', '-sf',
    default='data/string/protein.sequences.v10.5.fa.gz',
    help='Corpus with GO-Plus definition axioms')
def main(string_db_file, sequences_file):
    # Read interactions data
    mapping = {}
    data = []
    with gzip.open(string_db_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split()
            score = int(it[2])
            if score < 700:
                continue
            if it[0] not in mapping:
                mapping[it[0]] = len(mapping)
            if it[1] not in mapping:
                mapping[it[1]] = len(mapping)
            p1 = mapping[it[0]]
            p2 = mapping[it[1]]
            data.append((p1, p2))

    # Save data
    df = pd.DataFrame({'data': data})
    df.to_pickle('data/string/data.pkl')
    print('Saved interactions data')

    # Read sequences
    seqs = list()
    proteins = list()
    mappings = []
    seq = ''
    p_id = ''
    with gzip.open(sequences_file, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    if p_id in mapping:
                        seqs.append(to_ngrams(seq))
                        proteins.append(p_id)
                        mappings.append(mapping[p_id])
                    seq = ''
                space_ind = line.find(' ')
                p_id = line[1:space_ind]
            else:
                seq += line
    if p_id in mapping:
        seqs.append(to_ngrams(seq))
        proteins.append(p_id)
        mappings.append(mapping[p_id])

    # Save sequence data
    df = pd.DataFrame({'proteins': proteins, 'sequences': seqs, 'mappings': mappings})
    df.to_pickle('data/string/proteins.pkl')

    print('Saved proteins data')

    

                


if __name__ == '__main__':
    main()
