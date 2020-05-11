#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

rm -r $WORKDIR/MLScores
mkdir -p $WORKDIR/MLScores
python ml/apply.py $WORKDIR
