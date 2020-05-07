#!/bin/bash

# Select and create working directory
WORKDIR=$1
mkdir -p $WORKDIR

# Produce shapes
shapes/shapes.sh $WORKDIR

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

# Fit
fit/signal_strength.sh $WORKDIR
fit/scan.sh $WORKDIR
plot/scan.sh $WORKDIR
