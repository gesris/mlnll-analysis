#!/bin/bash

WORKDIR=$1

for INFLATION in 10
do 
    ## Create datacards
    echo "Creating Datacards"
    fit/convert2ch.sh $WORKDIR
    fit/datacards.sh $WORKDIR

    ## Manipulate datacards
    echo "Manipulating Datacards"
    sed -i '18,28d' $WORKDIR/cmb/125/htt_mt_0_2018.txt  #Delete all lines with splitted unc.
    DATACARD_LINE="scale_j_tot                     shape   1               1               -               1               1               1               1               1               1               1               1               1               1"
    echo $DATACARD_LINE >> $WORKDIR/cmb/125/htt_mt_0_2018.txt   #Append tot_unc entry to datacard

    ## Create tot_unc in ROOTfile
    echo "Manipulationg ROOTfile"
    ml/check_jes.sh $WORKDIR $INFLATION

    ## Continue analysis
    echo "Continuing Analysis"
    fit/workspace.sh $WORKDIR
    fit/workspace2shapes.sh $WORKDIR
    plot/analysis.sh $WORKDIR
    fit/signal_strength.sh $WORKDIR
    fit/scan.sh $WORKDIR
    plot/scan.sh $WORKDIR
    fit/significance.sh $WORKDIR

    ## Move scan.png to destined directory
    echo "Moving scan.png to webdir"
    #mv $WORKDIR/scan.png /etpwww/web/gristo/public_html/JES_scans/JES_totunc_trainnosys_x$INFLATION.png
done
