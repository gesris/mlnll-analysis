#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

rm -rf $WORKDIR/model_fold?
mkdir -p $WORKDIR/model_fold0
mkdir -p $WORKDIR/model_fold1
python3 ml/train.py $WORKDIR 0
python3 ml/train.py $WORKDIR 1
