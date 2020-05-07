#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

${CMSSW_BASE}/bin/slc7_amd64_gcc700/MorphingSMRun2Legacy \
    --base_path=$PWD \
    --input_folder_mt=$WORKDIR"/" \
    --real_data=false \
    --classic_bbb=true \
    --jetfakes=false \
    --embedding=false \
    --postfix="-ML" \
    --midfix="-NLL-" \
    --channel="mt" \
    --auto_rebin=false \
    --stxs_signals="stxs_stage0" \
    --categories="mlnll" \
    --era=2018 \
    --rebin_categories=false \
    --output=$WORKDIR"/" | tee ${WORKDIR}/datacards.log

exit
