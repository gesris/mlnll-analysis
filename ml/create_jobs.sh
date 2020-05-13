#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

rm -rf $WORKDIR/MLScores
mkdir -p $WORKDIR/MLScores

rm -rf $WORKDIR/MLScores_jobs
mkdir -p $WORKDIR/MLScores_jobs
mkdir -p $WORKDIR/MLScores_jobs/err
mkdir -p $WORKDIR/MLScores_jobs/log
mkdir -p $WORKDIR/MLScores_jobs/out

python ml/collect_jobs.py $WORKDIR
