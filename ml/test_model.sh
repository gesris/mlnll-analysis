#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

for FOLD in 0 1
do
    python ml/test_model.py $WORKDIR $FOLD
    #python ml/scan_crosscheck.py $WORKDIR $FOLD
done