#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_exp.pkl',
    help='Uniprot data file')
@ck.option(
    '--out-file', '-o', default='data/swissprot_exp.fa',
    help='swissprot proteins with exp annots')
def main(data_file, out_file):
    # Load interpro data
    df = pd.read_pickle(data_file)

    with open(out_file, 'w') as f:
        for row in df.itertuples():
            f.write('>' + row.proteins + '\n')
            f.write(row.sequences + '\n')
    

if __name__ == '__main__':
    main()
