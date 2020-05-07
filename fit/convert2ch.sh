#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python fit/convert2ch.py $WORKDIR
