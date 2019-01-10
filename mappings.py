#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd

from utils import GeneOntology

@ck.command()
@ck.option(
    '--go-file', '-gf', default='data/go.obo',
    help='Gene Ontology file in OBO Format')
@ck.option(
    '--corpus-file', '-cf', default='data/go_corpus_expanded.txt',
    help='Corpus with GO-Plus definition axioms')
@ck.option(
    '--out-file', '-o', default='data/mappings.txt',
    help='Result file with a list of terms for prediction task')
def main(go_file, corpus_file, out_file):
    go = GeneOntology(go_file, with_rels=True)

    w = open(out_file, 'w')
    with open(corpus_file, 'r') as f:
        for line in f:
            it = line.strip().split(': ')
            w.write(it[0])
            it = it[1].split(' and ')
            for x in it:
                x = x.split(' some ')
                x = x[0] if len(x) == 1 else x[1]
                w.write('\t' + x)
            w.write('\n')
    w.close()
            
                
    

                


if __name__ == '__main__':
    main()
