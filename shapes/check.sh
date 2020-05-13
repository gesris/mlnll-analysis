#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

python shapes/check.py $WORKDIR
