#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1

hadd -f $WORKDIR/shapes_main.root $WORKDIR/shapes_files/shapes_*.root
