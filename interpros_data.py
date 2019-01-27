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
    '--min-annots', '-ma', default=10,
    help='Uniprot data file')
@ck.option(
    '--out-file', '-o', default='data/interpros.pkl',
    help='Result file with list of intepro ids')
def main(data_file, min_annots, out_file):
    # Load interpro data
    df = pd.read_pickle(data_file)
    cnt = Counter()
    for row in df.itertuples():
        for ipro in row.interpros:
            cnt[ipro] += 1
    interpros = list()
    for ipro, cnt in cnt.items():
        if cnt >= min_annots:
            interpros.append(ipro)
    df = pd.DataFrame({'interpros': interpros})
    logging.info(f'Saving {len(interpros)} InterPro ids')
    df.to_pickle(out_file)


if __name__ == '__main__':
    main()
