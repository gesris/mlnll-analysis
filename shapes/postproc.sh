#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python shapes/qcd_estimation.py $WORKDIR
python shapes/convert2ch.py $WORKDIR
