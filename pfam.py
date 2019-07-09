#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import Popen, PIPE
import time
from utils import Ontology
from aminoacids import to_onehot
import io
import gzip

MAXLEN = 2000

@ck.command()
@ck.option('--pfam-file', '-pf', default='data/Pfam-A.seed', help='Pfam')
def main(pfam_file):
    # Load CNN model
    pfams = list()
    alignments = list()
    interpros = list()
    with io.open(pfam_file, 'rt', encoding='latin-1') as f:
        pfam_id = ''
        aligns = list()
        ipro_id = ''
        for line in f:
            line = line.strip()
            if line.startswith('#=GF AC   '):
                if pfam_id != '':
                    pfams.append(pfam_id)
                    alignments.append(aligns)
                    interpros.append(ipro_id)
                    aligns = list()
                    ipros = list()
                pfam_id = line[10:]
            elif line.startswith('#=GF DR   INTERPRO;'):
                ipro_id = line[20:-1]
            elif not line.startswith('#') and not line == '//':
                a = line.split()
                if len(a) == 2:
                    aligns.append(a)
                else:
                    print(a)

    with open('data/pfam.fa', 'w') as w:
        for pfam_id, aligns, ipro_id in zip(pfams, alignments, interpros):
            for a in aligns:
                w.write(f'>{pfam_id}; {ipro_id}; {a[0]}\n')
                w.write(a[1].replace('.','') + '\n')
            

if __name__ == '__main__':
    main()
