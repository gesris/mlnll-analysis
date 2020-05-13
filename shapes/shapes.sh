#!/bin/bash

# TODO: Use LCG here
source utils/setup_lcg.sh

WORKDIR=$1

python shapes/shapes.py $WORKDIR
