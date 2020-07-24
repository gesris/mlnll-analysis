#!/bin/bash

source ../utils/setup_lcg.sh

WORKDIR=$1

python plot/histogram.py $WORKDIR '100'