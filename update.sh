#!/bin/bash


export LD_LIBRARY_PATH=/usr/lib/cuda/include:/usr/lib/cuda/lib64:$LD_LIBRARY_PATH

export PATH=/home/deepgo/bin:$PATH

cd /data/deepgo/deepgoplus_test
source venv/bin/activate
python update.py
