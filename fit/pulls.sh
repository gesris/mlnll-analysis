
#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

pushd $WORKDIR

combine -M FitDiagnostics \
        -n .Pulls \
        -m 125 \
        -d workspace.root \
        --robustFit 1 | tee pulls_fit.log

python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a -f html fitDiagnostics.Pulls.root > pulls.html
python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a -f text fitDiagnostics.Pulls.root > pulls.txt
