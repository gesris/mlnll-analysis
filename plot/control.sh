#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python plot/control.py $WORKDIR
