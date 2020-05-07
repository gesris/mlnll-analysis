#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

pushd $WORKDIR
combine -M FitDiagnostics \
        -n .MLNLL \
        -m 125 \
        -d workspace.root \
        --robustFit 1 | tee signal_strength.log
