#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

pushd $WORKDIR
combine -M Significance \
        -m 125 \
        -d workspace.root | tee significance.log
