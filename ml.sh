#!/bin/bash

# Select and create working directory
WORKDIR=$1
mkdir -p $WORKDIR

# Create training dataset
echo "  Create Training Dataset"
#ml/dataset.sh $WORKDIR

# Train model
echo "  Train Model"
ml/train.sh $WORKDIR

# Validate model
echo "  Validate Model"
ml/test_model.sh $WORKDIR
