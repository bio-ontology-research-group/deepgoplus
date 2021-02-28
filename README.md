# DeepGOPlus: Improved protein function prediction from sequence

DeepGOPlus is a novel method for predicting protein functions from
protein sequences using deep neural networks combined with sequence
similarity based predictions.

This repository contains script which were used to build and train the
DeepGOPlus model together with the scripts for evaluating the model's
performance.

## Dependencies
* The code was developed and tested using python 3.6.
* To install python dependencies run:
  `pip install -r requirements.txt`
* Install [diamond](https://github.com/bbuchfink/diamond) program on your system (diamond command should be available)


## Data
* http://deepgoplus.bio2vec.net/data/ - Here you can find the data
used to train and evaluate our method.
 * data.tar.gz - Data required to run predict.sh script
 * data-cafa.tar.gz - CAFA3 challenge dataset
 * data-2016.tar.gz - Dataset which is used to compare DeepGOPlus with
   GOLabeler and DeepText2GO

## Installation
`pip install deepgoplus`

## Running
* Download all the files from http://deepgoplus.bio2vec.net/data/data.tar.gz and place them into data folder
* `deepgoplus --data-root <path_to_data_folder> --in-file <input_fasta_filename>`


## Scripts
The scripts require GeneOntology in OBO Format.
* uni2pandas.py - This script is used to convert data from UniProt
database format to pandas dataframe.
* deepgoplus_data.py - This script is used to generate training and
  testing datasets.
* deepgoplus.py - This script is used to train the model
* evaluate_*.py - The scripts are used to compute Fmax, Smin and AUPR

The online version of DeepGOPlus is available at http://deepgoplus.bio2vec.net/

## Citation

If you use DeepGOPlus for your research, or incorporate our learning algorithms in your work, please cite:
Maxat Kulmanov, Robert Hoehndorf; DeepGOPlus: Improved protein function prediction from sequence, Bioinformatics, https://doi.org/10.1093/bioinformatics/btz595



## New version specifications
Current dependencies can be found in the requirements.txt file.
The used Python version is 3.7.9.
Current version of Tensorflow will require Cuda 10.1 and Cudnn 7.6.5


## Updating

The following scripts must be run to update the model using the latest versions of the Gene Ontology (GO) and the SwissProt Database.

* update.py - This will download new releases of GO and SwissProt and train the model. If there are not new releases, the process will abort.
* new_evaluation.sh - This will compute Fmax, Smin and AUPR metrics. 
