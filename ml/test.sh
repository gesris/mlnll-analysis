#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

for FOLD in 0 1
do
    python ml/test_confusion.py $WORKDIR $FOLD
    python ml/test_taylor.py $WORKDIR $FOLD
done
