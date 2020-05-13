#!/bin/bash

set -e

source utils/setup_lcg.sh

WORKDIR=$1

OUTPUTDIR=$WORKDIR/MLScores

for DIR in $(ls $OUTPUTDIR)
do
    echo "Merge files in directory" $DIR
    pushd $OUTPUTDIR/$DIR
    hadd -f ${DIR}.root mt_*.root
    popd
done
