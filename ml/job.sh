#!/bin/bash

# Exit on error
set -e

echo ">>> Begin job"

JOBID=$1
echo "JOBID:" $JOBID

SRCDIR=$2
echo "SRCDIR:" $SRCDIR

WORKDIR=$3
echo "WORKDIR:" $WORKDIR

FOLDER=$4
echo "FOLDER:" $FOLDER

FILENAME=$5
echo "FILENAME:" $FILENAME

echo ">>> Start working"

echo "Trigger auto mount of CVMFS"
ls /cvmfs
ls /cvmfs/sft.cern.ch

cd $SRCDIR
echo "PWD:" $PWD
source utils/setup_lcg.sh

echo "USER: " && whoami
echo "PYTHONPATH: " $PYTHONPATH

python3.6 ml/job.py $WORKDIR $FOLDER $FILENAME

echo ">>> End job"
