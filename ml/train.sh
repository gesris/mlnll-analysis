#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python ml/train.py $WORKDIR 0
python ml/train.py $WORKDIR 1
