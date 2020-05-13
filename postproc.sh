#!/bin/bash

WORKDIR=$1

# Postprocess shapes for process estimations
shapes/postproc.sh $WORKDIR

# Control plots
plot/control.sh $WORKDIR

# Create datacards and workspacd
fit/convert2ch.sh $WORKDIR
fit/datacards.sh $WORKDIR
fit/workspace.sh $WORKDIR

# Prefit plots
fit/workspace2shapes.sh $WORKDIR
plot/analysis.sh $WORKDIR

# Signal strenght
fit/signal_strength.sh $WORKDIR

# NLL scan
fit/scan.sh $WORKDIR
plot/scan.sh $WORKDIR

# Significance
fit/significance.sh $WORKDIR
