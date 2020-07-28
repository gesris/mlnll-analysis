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

echo ">>> Start working"

echo "Trigger auto mount of CVMFS"
ls /cvmfs
ls /cvmfs/sft.cern.ch

echo "Trigger auto mount of ceph"
ls /ceph
ls /ceph/htautau

cd $SRCDIR
echo "PWD:" $PWD

source utils/setup_lcg.sh

python shapes/job.py $WORKDIR $JOBID

echo ">>> End job"
