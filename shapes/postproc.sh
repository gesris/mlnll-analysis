#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python shapes/remove_split.py $WORKDIR
python shapes/qcd_estimation.py $WORKDIR
