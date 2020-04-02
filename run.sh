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
