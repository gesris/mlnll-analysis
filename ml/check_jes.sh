#!/bin/bash

source utils/setup_lcg.sh

WORKDIR=$1
INFLATION=$2
ROOTFILE="/cmb/common/htt_input_2018.root"

python ml/check_jes.py $WORKDIR$ROOTFILE $INFLATION
