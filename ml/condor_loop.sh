#!/bin/bash

# While loop counting sum of elements in directory for automatization

WORKDIR=$1

run=true
step=0

while $run; do
    sleep 120s
    clear
    if [ $(ls -l /work/gristo/mlnll-analysis/$WORKDIR/MLScores_jobs/err | grep err | wc -l) =  1126 ]; then
        printf "\n\n## --------------  All jobs done  --------------- ##\n\n"
        run=false
        break
    fi

    if [ $(condor_q | grep HOLD | wc -l) = 1 ]; then
        printf "\n\n## ---------  Releasing jobs from HOLD  --------- ##\n\n"
        sh /work/gristo/mlnll-analysis/condor_release.sh
    fi

    condor_q | head -5

    # Breakup sequence after too many loops
    #step=$(($step + 1))
    #if [ $step = 1000]
done
