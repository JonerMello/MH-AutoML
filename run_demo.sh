#!/bin/bash
cd src
python3.8 -m pip install .
python3.8 view/main.py -d ../Datasets/dataset_sujo.csv -l class

