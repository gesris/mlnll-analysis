#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python ml/check.py $WORKDIR
