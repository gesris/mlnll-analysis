#!/bin/bash

WORKDIR='/home/gristo/mlnll-analysis/output/4_bins_nosys_noclassw_2'

git commit -m "bugfixes" ml/scan_cross_check.py
git push origin tf_scan_cross_check

ssh gristo@bms3 'cd /home/gristo/mlnll-analysis && git pull'

for FOLD in 0 1
do
    ssh gristo@bms3 'cd /home/gristo/mlnll-analysis && source utils/setup_lcg.sh && python ml/scan_cross_check.py' $WORKDIR $FOLD
    scp bms3:$WORKDIR/model_fold$FOLD/scan_cross_check.png /home/risto/Masterarbeit/mlnll-analysis/plots/test/4_bins_nosys_noclassw_scan_cross_check_fold$FOLD.png
done

