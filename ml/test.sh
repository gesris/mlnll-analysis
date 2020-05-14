#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

#python ml/test_confusion.py $WORKDIR 0
#python ml/test_confusion.py $WORKDIR 1

python ml/test_taylor.py $WORKDIR 0
