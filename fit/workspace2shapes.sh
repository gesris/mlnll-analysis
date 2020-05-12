#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1
pushd $WORKDIR

# Prefit shapes
PostFitShapesFromWorkspace -m 125 \
    -w workspace.root \
    -d cmb/125/combined.txt.cmb \
    -o shapes_prefit.root | tee workspace2shapes_prefit.log

# ML fit for the postfit
combine -M FitDiagnostics \
        -n .PrefitPostfit \
        -m 125 \
        -d workspace.root \
        --robustFit 1 -v1 \
        --cminDefaultMinimizerStrategy 0 | tee workspace2shapes_mlfit.log

# Prefit and postfit shapes
PostFitShapesFromWorkspace -m 125 \
    -w workspace.root \
    -d cmb/125/combined.txt.cmb \
    -f fitDiagnostics.PrefitPostfit.root:fit_s \
    -o shapes_postfit.root --postfit | tee workspace2shapes_postfit.log
