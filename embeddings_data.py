#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip

from utils import Ontology, read_fasta
from aminoacids import MAXLEN, to_ngrams

@ck.command()
@ck.option(
    '--vectors-file', '-wf', default='data/string/vectors.txt',
    help='Word2Vec output file')
@ck.option(
    '--mapping-file', '-wf', default='data/string/graph_mapping.out',
    help='gen_graph.py mapping file')
@ck.option(
    '--sequences-file', '-sf',
    default='data/string/protein.sequences.v10.fa.gz',
    help='Corpus with GO-Plus definition axioms')
def main(vectors_file, mapping_file, sequences_file):
    # Read mapping
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            mapping[int(it[1])] = it[0]
    
    # Read vectors
    vectors = {}
    with open(vectors_file, 'r') as f:
        next(f)
        next(f)
        for line in f:
            it = line.strip().split()
            p_id = mapping[int(it[0])]
            vector = np.array(list(map(float, it[1:])), dtype='float32')
            vectors[p_id] = vector
    
            
    # Read sequences
    seqs = []
    proteins = []
    embeddings = []
    seq = ''
    p_id = ''
    with gzip.open(sequences_file, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    if p_id in vectors:
                        seqs.append(seq)
                        proteins.append(p_id)
                        embeddings.append(vectors[p_id])
                    seq = ''
                space_ind = line.find(' ')
                p_id = line[1:space_ind]
            else:
                seq += line
    if p_id in vectors:
        seqs.append(seq)
        proteins.append(p_id)
        embeddings.append(vectors[p_id])

    # Save sequence data
    df = pd.DataFrame({'proteins': proteins, 'sequences': seqs, 'embeddings': embeddings})
    df.to_pickle('data/string/embeddings.pkl')

    print('Saved proteins data')

    

                


if __name__ == '__main__':
    main()
