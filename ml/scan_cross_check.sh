#!/bin/bash

source ../utils/setup_lcg.sh

WORKDIR=$1

for FOLD in 0 1
do
    python scan_cross_check.py $WORKDIR $FOLD
done

