#!/usr/bin/env python

import os
import sys
import logging
import subprocess
from datetime import datetime
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
        
        last_release_version = last_release_data["current_uniprot_version"]

        print(f'Checking new release date...')

        response = requests.get('https://www.uniprot.org')

        new_release_version = response.headers['X-UniProt-Release']
        
        if last_release_version == new_release_version:
            print(f'There are not new releases\nAborting...')
            new_release = False
        else:
            print(f'SwissProt release version ' + new_release_version + ' found...')

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

            last_release_data["current_uniprot_version"] = new_release_version


       	    ## Update version info
            curr_version = last_release_data["version"].split('.')
            prefix = curr_version[:-1]
            suffix = curr_version[-1]
            last_release_data["version"] = ".".join(prefix + [str(int(suffix)+1)])


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
    cmd = ["python", "deepgoplus.py"]
    proc = subprocess.run(cmd)




###################################################################################################


def compress_data():


    with open(last_release_metadata, 'r') as f:
        last_release_data = json.load(f)
    version = last_release_data["version"]

    #date = datetime.strptime(last_release_data["current_date"], '%a, %d %b %Y %H:%M:%S GMT')
    out_file = "data/data-" + version + '.tar.gz'

    go_file="data/go.obo"
    diamond_db="data/train_data.dmnd"
    model="data/model.h5"
    result_diamond="data/test_diamond.res"
    predictions="data/predictions.pkl"
    train_pkl="data/train_data.pkl"
    train_fa="data/train_data.fa"
    test_pkl="data/test_data.pkl"
    test_fa="data/test_data.fa"

    terms="data/terms.pkl"

    release="data/RELEASE.md"
    release_html="data/RELEASE.html"

    metadata = "metadata/last_release.json"


    cmd = ["tar", "-czf", out_file, go_file, diamond_db, model, result_diamond, predictions, train_pkl, train_fa, test_pkl, test_fa, terms, release, release_html, metadata]
    proc = subprocess.run(cmd)

    return out_file[5:]


def metrics_from_files():

    cmd = ["bash", "new_evaluation.sh"]
    proc = subprocess.run(cmd)

    mf = open('results/deepgoplus_mf.txt').readlines()
    bp = open('results/deepgoplus_bp.txt').readlines()
    cc = open('results/deepgoplus_cc.txt').readlines()

    mf_smin = mf[2].split(':')[1]
    mf_fmax = mf[3].split(':')[1]
    mf_aupr = mf[4].split(':')[1]

    bp_smin = bp[2].split(':')[1]
    bp_fmax = bp[3].split(':')[1]
    bp_aupr = bp[4].split(':')[1]

    cc_smin = cc[2].split(':')[1]
    cc_fmax = cc[3].split(':')[1]
    cc_aupr = cc[4].split(':')[1]

    return mf_smin, mf_fmax, mf_aupr, bp_smin, bp_fmax, bp_aupr, cc_smin, cc_fmax, cc_aupr


def release_notes_file():
    # Will generate data/RELEASE.md and data/RELEASE.html files.

    with open(last_release_metadata, 'r') as f:
            last_release_data = json.load(f)

    version = last_release_data["version"]

    file_html = open('data/RELEASE.html', 'w')
    file_md = open('data/RELEASE.md', 'w')
    go_file = open('data/go.obo', 'r')
    mf_smin, mf_fmax, mf_aupr, bp_smin, bp_fmax, bp_aupr, cc_smin, cc_fmax, cc_aupr = metrics_from_files()

    go_file.readline()
    go_date =  str(datetime.strptime(go_file.readline().rstrip('\n').split('/')[1], '%Y-%m-%d').date())
    swissprot_version = last_release_data["current_uniprot_version"]

   
    text_md = """
# DeepGOPlus: Improved protein function prediction from sequence
DeepGOPlus is a novel method for predicting protein functions from
protein sequences using deep neural networks combined with sequence
similarity based predictions.
# Release information
Current version is """ + version + """. The model in the current release was trained using the Gene Ontology
released on """ + go_date + """ and the SwissProt data with version """ + swissprot_version +  """.
The obtained results are the following:
For MFO:
    Fmax:   """ + mf_fmax +"""
    Smin:   """ + mf_smin +"""
    AUPR:   """ + mf_aupr +"""
For BPO:
    Fmax:   """ + bp_fmax + """
    Smin:   """ + bp_smin + """
    AUPR:   """ + bp_aupr + """
For CCO:
    Fmax:   """ + cc_fmax + """ 
    Smin:   """ + cc_smin + """    
    AUPR:   """ + cc_aupr + """
For more information about the project, please look at https://github.com/bio-ontology-research-group/deepgoplus 
"""

    text_html = """
    
<p>
The current version release is """ + version + """. The model in the current release was trained using the Gene Ontology
# released on """ + go_date + """ and the SwissProt data released on """ + swissprot_version +""".
</p>

<p>
The obtained results are the following:
</p>

<table class="table table-striped">
<thead>
<tr><th></th>
<th>Fmax</th>
<th>Smin</th>
<th>AUPR</th>
</tr></thead>
<tbody>
<tr><td>MFO</td><td>""" + mf_fmax + """</td><td>""" + mf_smin+ """</td><td>""" + mf_aupr+"""</td></tr>
<tr><td>BPO</td><td>""" + bp_fmax + """</td><td>""" + bp_smin+ """</td><td>""" + bp_aupr+"""</td></tr>
<tr><td>CCO</td><td>""" + cc_fmax + """</td><td>""" + cc_smin+ """</td><td>""" + cc_aupr +"""</td></tr>
</tbody>
</table>
    
"""

    file_md.write(text_md)
    file_html.write(text_html)




def upload_data(filename):
    

    deepgo_server_ip = "10.254.146.187"
    login_cmd = "zhapacfp@"+deepgo_server_ip
    
    

    cmd = ["ssh", login_cmd, "/home/zhapacfp/bin/take_data_ontolinator", filename]
    proc = subprocess.run(cmd)










def main():
    downloaded = download_data()
    if downloaded:
        prepare_data()
        train_data()

        release_notes_file()
        out_file_name = compress_data() #compress the data and return the name(string) of the file

        upload_data(out_file_name)

if __name__ == "__main__":
    main()
