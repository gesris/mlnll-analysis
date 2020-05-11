#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

# Collect required events and write out as ROOT files
rm -f $WORKDIR/*fold*.root
python ml/collect_events.py $WORKDIR

# Merge to unified datasets
hadd -f $WORKDIR/fold0.root $WORKDIR/*fold0.root
hadd -f $WORKDIR/fold1.root $WORKDIR/*fold1.root
