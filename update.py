#!/usr/bin/env python

import os
import sys
import logging
import subprocess
import wget
import requests
import json

# MAXLEN = 2000

data_root = 'data/'
last_release_metadata = 'metadata/last_release.json' 

def download_data():
    # Downloads data from The Gene Ontology (updated monthly)
    # Downloads data from Uniprot/Swissprot (updated every two months)

    new_release = True

    try:
        if not os.path.exists(data_root):
            os.mkdir(data_root, 0o755,)


        with open(last_release_metadata, 'r') as f:
            last_release_data = json.load(f)
        
        last_release_date = last_release_data["current_date"]

        print(f'Checking new release date...')

        response = requests.get('https://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/reldate.txt')

        new_release_date = response.headers['Last-Modified']

        if last_release_date == new_release_date:
            print(f'There are not new releases\nAborting...')
            new_release = False
        else:
            print(f'There was a new release of SwissProt file on ' + new_release_date)

            print(f'Proceeding to download Gene Ontology file go.obo from http://purl.obolibrary.org/obo/go.obo')
            urlGO = 'http://purl.obolibrary.org/obo/go.obo'
             
            cmd = ["rm", "data/go.obo"]
            proc = subprocess.run(cmd)
            wget.download(urlGO, out = 'data/go.obo')
        
        
            print(f'\nProceeding to download SwissProt file')
            urlSwiss = 'ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'

            cmd = ["rm", "data/uniprot_sprot.dat.gz"]
            proc = subprocess.run(cmd)
            wget.download(urlSwiss, out ='data/uniprot_sprot.dat.gz')

            cmd = ["mv", 'data/swissprot.pkl', 'data/old_swissprot.pkl']
            proc = subprocess.run(cmd)

            last_release_data["previous_date"] = last_release_date
            last_release_data["current_date"] = new_release_date
        
            with open(last_release_metadata, 'w') as f:
                json.dump(last_release_data, f)
      


    except Exception as e:
        logging.error(e)
        sys.exit(1)

    return new_release


def prepare_data():
    #Prepares the necessary data to train the model

    print(f'Preparing data...')

    #get new swissprot.pkl through uni2pandas.py
    cmd = ["python", "uni2pandas.py"]
    proc = subprocess.run(cmd)

    #get terms.pkl, train_data.pkl and test_data.pkl through deepgoplus_data,py
    cmd = ["python", "deepgoplus_data.py"]
    proc = subprocess.run(cmd)

def train_data():
    #Trains the data by running deepgoplus.py

    print(f'Sending job to IBEX...')

    cmd = ["sbatch", "jobTrain"]
    proc = subprocess.run(cmd)

def main():
    downloaded = download_data()
    if downloaded:
        prepare_data()
        train_data()

if __name__ == "__main__":
    main()
