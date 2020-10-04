#!/usr/bin/env python

# import click as ck
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from subprocess import Popen, PIPE
# import time
# from deepgoplus.utils import Ontology, NAMESPACES
# from deepgoplus.aminoacids import to_onehot
# import gzip
import os
import sys
import logging
import subprocess
import wget

# MAXLEN = 2000

data_root = 'data/'

def download_data():
    # Downloads data from The Gene Ontology (updated monthly)
    # Downloads data from Uniprot/Swissprot (updated every two months)
    try:
        if not os.path.exists(data_root):
            os.mkdir(data_root, 0o755,)
        print(f'Proceeding to download Gene Ontology file go.obo from http://purl.obolibrary.org/obo/go.obo')
        urlGO = 'http://purl.obolibrary.org/obo/go.obo'
        wget.download(urlGO, 'data/go.obo')

        print(f'\nProceeding to download SwissProt file')
        urlSwiss = 'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'
        wget.download(urlSwiss, 'data/swissprot.dat.gz')
    except Exception as e:
        logging.error(e)
        sys.exit(1)


def prepare_data():
    #Prepares the necessary data to train the model

    #get new swissprot.pkl through uni2pandas.py
    cmd = ["python", "uni2pandas.py"]
    proc = subprocess.run(cmd)

    #get terms.pkl, train_data.pkl and test_data.pkl through deepgoplus_data,py
    cmd = ["python", "deepgoplus_data.py"]
    proc = subprocess.run(cmd)

def train_data():
    #Trains the data by running deepgoplus.py
    cmd = ["sbatch", "jobTrain"]
    proc = subprocess.run(cmd)

def main():
    download_data()
    prepare_data()
    train_data()

if __name__ == "__main__":
    main()
