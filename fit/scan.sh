#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

pushd $WORKDIR
combineTool.py -M MultiDimFit \
    -d workspace.root \
    -m 125 \
    --algo grid \
    -P r \
    --floatOtherPOIs 1 \
    --points 30 \
    --setParameterRanges r=0,2 \
    -n .Scan
