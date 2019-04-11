#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import math
import os

logging.basicConfig(level=logging.INFO)

@ck.command()
def main():
    # SLURM JOB ARRAY INDEX
    kernels = [33, 65, 129, 257, 513]
    dense_depths = [0, 1, 2]
    nb_filters = [32, 64, 128, 256, 512]
    missings = []
    for pi in range(75):
        params_index = pi
        max_kernel = kernels[pi % 5] - 1
        pi //= 5
        dense_depth = dense_depths[pi % 3] + 1
        pi //= 3
        nb_filter = nb_filters[pi % 5]
        pi //= 5
        job_file = f'job_{params_index}.out'
        if not os.path.exists(job_file):
            continue
        with open(job_file) as f:
            val_loss = 'missing'
            test_loss = 'missing'
            for line in f:
                line = line.strip()
                s = 'val_loss did not improve from '
                ind = line.find(s)
                if ind != -1:
                    val_loss = line[ind + len(s):]
                s = 'Test loss '
                ind = line.find(s)
                if ind != -1:
                    test_loss = line[ind + len(s):]
                
        print(f'{params_index + 1} & {max_kernel} & {dense_depth} & {nb_filter} & {val_loss} & {test_loss} \\\\')
        if test_loss == 'missing':
            missings.append(str(params_index))
    print(' '.join(missings))
        
if __name__ == '__main__':
    main()
