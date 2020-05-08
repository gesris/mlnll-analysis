#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python ml/test.py $WORKDIR 0
python ml/test.py $WORKDIR 1
