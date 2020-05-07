#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

${CMSSW_BASE}/bin/slc7_amd64_gcc700/MorphingSMRun2Legacy \
    --base_path=$PWD \
    --input_folder_mt="output/" \
    --real_data=false \
    --classic_bbb=false \
    --binomial_bbb=true \
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
    --output="output/" | tee ${WORKDIR}/datacards.log

exit

# Use Barlow-Beeston-lite approach for bin-by-bin systematics
pushd ${OUTPUTDIR}/cmb/125/
for FILE in *.txt
do
    sed -i '$s/$/\n * autoMCStats 0.0/' ${FILE}
done
popd
