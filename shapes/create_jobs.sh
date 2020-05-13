#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1
mkdir -p $WORKDIR

python shapes/create_graphs.py $WORKDIR

rm -rf $WORKDIR/shapes_jobs
mkdir -p $WORKDIR/shapes_jobs
mkdir -p $WORKDIR/shapes_jobs/err
mkdir -p $WORKDIR/shapes_jobs/log
mkdir -p $WORKDIR/shapes_jobs/out
rm -rf $WORKDIR/shapes_files
mkdir -p $WORKDIR/shapes_files
python shapes/create_jobs.py $WORKDIR
