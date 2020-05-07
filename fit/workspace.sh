#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

combineTool.py -M T2W \
    -o $PWD/$WORKDIR/workspace.root \
    -i $WORKDIR/cmb/125 \
    --parallel 8 | tee $WORKDIR/workspace.log
