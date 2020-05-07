#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

pushd $WORKDIR
plot1DScan.py \
    higgsCombine.Scan.MultiDimFit.mH125.root \
    --POI r \
    --output scan
