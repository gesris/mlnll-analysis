#!/bin/bash

# Select and create working directory
WORKDIR=$1
#mkdir -p $WORKDIR

# Create training dataset
#ml/dataset.sh $WORKDIR

# Train model
ml/train.sh $WORKDIR

# Validate model
# TODO: Missing any validation!
ml/test_model.sh $WORKDIR
