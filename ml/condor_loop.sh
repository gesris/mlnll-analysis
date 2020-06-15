#!/bin/bash

# While loop counting sum of elements in directory for automatization

run=true
step=0

while $run; do
    sleep 120s
    if [ $(ls -l /work/gristo/mlnll-analysis/output/run7/MLScores_jobs/err | grep err | wc -l) =  1126]; then
        echo "## ------------ All jobs done ------------- ##"
        run=false
    fi

    if [ $(condor_q | grep HOLD | wc -l) = 1 ]; then
        echo "## ------- Releasing jobs from HOLD ------- ##"
        sh /work/gristo/mlnll-analysis/condor_release.sh
    fi

    condor_q

    # Breakup sequence after too many loops
    #step=$(($step + 1))
    #if [ $step = 1000]
done
