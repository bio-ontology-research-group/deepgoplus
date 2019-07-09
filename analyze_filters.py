#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from subprocess import Popen, PIPE
import time
from utils import Ontology
from aminoacids import to_onehot

MAXLEN = 2000

@ck.command()
@ck.option(
    '--filters-file', '-ff', default='data/filters.txt',
    help='File with filters')
@ck.option(
    '--interpro-file', '-if', default='data/Pfam-A.seed',
    help='InterPRO domain sequences')
def main(filters_file):
    filters = open(filters_file).read().splitlines()
    

if __name__ == '__main__':
    main()
