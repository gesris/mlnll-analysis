#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

# Prefit shapes
pushd $WORKDIR
PostFitShapesFromWorkspace -m 125 \
    -w workspace.root \
    -d cmb/125/combined.txt.cmb \
    -o shapes_prefit.root
