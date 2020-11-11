#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J zcpfJobUpdate
#SBATCH -o zcpfJobUpdate.%J.out
#SBATCH -e zcpfJobUpdate.%J.err
#SBATCH --mail-user=fernando.zhapacamacho@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
#SBATCH --mem=16G

#run the application:
python update.py
