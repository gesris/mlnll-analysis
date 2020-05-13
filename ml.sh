#!/bin/bash

# Select and create working directory
WORKDIR=$1
mkdir -p $WORKDIR

# Create training dataset
ml/dataset.sh $WORKDIR

# Train model
ml/train.sh $WORKDIR

# Validate model
ml/test.sh $WORKDIR

# Create jobs with model application
#ml/create_jobs.sh $WORKDIR

# Check model application and eventually run remaining jobs locally
#ml/check.sh $WORKDIR

# Merge application files
#ml/merge.sh $WORKDIR
