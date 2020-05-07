#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python plot/analysis.py $WORKDIR
