
#!/bin/bash

source utils/setup_cmssw.sh

WORKDIR=$1

pushd $WORKDIR

combineTool.py -M Impacts -m 125 -d workspace.root \
    --doInitialFit --robustFit 1 \
    -t -1 --expectSignal=1 \
    --parallel 12

combineTool.py -M Impacts -m 125 -d workspace.root \
    --doFits --robustFit 1 \
    -t -1 --expectSignal=1 \
    --parallel 12

combineTool.py -M Impacts -m 125 -d workspace.root --output impacts.json
plotImpacts.py -i impacts.json -o impacts
