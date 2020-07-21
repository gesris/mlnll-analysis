#!/bin/bash

# While loop counting sum of elements in directory for automatization

WORKDIR=$1
ANALYSISDIR=$2

run=true
step=0

while $run; do
    sleep 120s
    if [ $(ls -l $ANALYSISDIR/$WORKDIR/MLScores_jobs/err | grep err | wc -l) =  1126 ]; then
        printf "\n\n## --------------  All jobs done  --------------- ##\n\n"
        run=false
        break
    fi

    elif [ $(condor_q | grep HOLD | wc -l) = 1 ]; then
        printf "\n\n## ---------  Releasing jobs from HOLD  --------- ##\n\n"
        sh $ANALYSISDIR/condor_release.sh
    fi

    condor_q
    printf "_______________________________________________________________________________________________________________\n"

    # Breakup sequence after too many loops
    #step=$(($step + 1))
    #if [ $step = 1000]
done
