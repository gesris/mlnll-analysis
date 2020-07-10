#!/bin/bash


printf "\n\n## ---------------------------------------------- ##\n##                       ___            ___       ##\n##                      /   \          /   \      ##\n##                      \_   \        /  __/      ##\n##                       _\   \      /  /__       ##\n##                       \___  \____/   __/       ##\n##                           \_       _/          ##\n##                             | O O  \_          ##\n##         RUN STARTED,        |                  ##\n##          GOOD LUCK!       _/     /\            ##\n##                          /o)  (o/\ \_          ##\n##                          \_____/ /             ##\n##                            \____/              ##\n## ---------------------------------------------- ##\n"


## Commiting latest changes and pushing to origin
printf "\n\n## ---------    Committing changes      --------- ##\n\n"

#git commit -m " automatically generated arbitrary commit message for convenience" .
#git push origin nll_loss_implementation_test


## Setting up directories
## Always change RUN and ANALYSISDIR for each run
printf "\n\n## ---------    Setting up directories  --------- ##\n\n"

RUN='4_bins_nosys_noclassw'
ANALYSISDIR='/home/gristo/mlnll-analysis'
WORKDIR='output/'$RUN

DIR=$(date +'%Y_%m_%d')
LOCALDIR='/home/risto/Masterarbeit/mlnll-analysis/plots/'$DIR
REMOTEDIR='/etpwww/web/gristo/public_html/'$DIR

#mkdir $LOCALDIR
#mkdir $LOCALDIR/$RUN
#ssh -t -n gristo@bms3 'mkdir '$ANALYSISDIR/$WORKDIR
#ssh -t -n gristo@bms3 'mkdir '$ANALYSISDIR/$WORKDIR'_shapes'
#ssh -t -n gristo@bms3 'mkdir '$REMOTEDIR/
#ssh -t -n gristo@bms3 'mkdir '$REMOTEDIR/$RUN
#ssh -t -n gristo@bms3 'cp '$REMOTEDIR'/../index.php '$REMOTEDIR/
#ssh -t -n gristo@bms3 'cp '$REMOTEDIR'/../index.php '$REMOTEDIR/$RUN/


## Pull latest commits from origin
printf "\n\n## --------    Pulling latest commits    -------- ##\n\n"

#ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && git pull'


## Run ML part
printf "\n\n## -----------    Starting ML part    ----------- ##\n\n"

ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/dataset.sh '$WORKDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/train.sh '$WORKDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/test_model.sh '$WORKDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/create_jobs.sh '$WORKDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR/$WORKDIR'/MLScores_jobs/ && condor_submit job.jdl'
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/condor_loop.sh '$WORKDIR $ANALYSISDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/check.sh '$WORKDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh ml/merge.sh '$WORKDIR

for FOLD in 0 1
do
        ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && mv '$WORKDIR'/model_fold'$FOLD'/histogram.png '$WORKDIR'_shapes/ && mv '$WORKDIR'/model_fold'$FOLD'/minimization.png '$WORKDIR'_shapes/'
done

## Run SHAPES part
printf "\n\n## ---------    Starting SHAPES part    --------- ##\n\n"

ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh shapes/create_jobs.sh '$WORKDIR'_shapes'
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR/$WORKDIR'_shapes/shapes_jobs/ && condor_submit job.jdl'
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh shapes/condor_loop.sh '$WORKDIR'_shapes' $ANALYSISDIR
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh shapes/check.sh '$WORKDIR'_shapes'
ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh shapes/merge.sh '$WORKDIR'_shapes'


## Make graphs
printf "\n\n## ------------    Making graphs    ------------- ##\n\n"

ssh -t -n gristo@bms3 'cd '$ANALYSISDIR' && sh postproc.sh '$WORKDIR'_shapes'


## Copy to remote WEBSERVER
printf "\n\n## -------    Copying to WEB-directory    ------- ##\n\n"

ssh -t -n gristo@bms3 'cp '$ANALYSISDIR/$WORKDIR'_shapes/*.png '$REMOTEDIR/$RUN/
ssh -t -n gristo@bms3 'cp '$ANALYSISDIR/$WORKDIR'_shapes/*.pdf '$REMOTEDIR/$RUN/


## Copy to local repo
printf "\n\n## ------    Copying to local directory    ------ ##\n\n"

scp bms3:$ANALYSISDIR/$WORKDIR'_shapes/'*.png $LOCALDIR/$RUN/
scp bms3:$ANALYSISDIR/$WORKDIR'_shapes/'*.pdf $LOCALDIR/$RUN/


printf "\n\n\n     ####################################################\
        \n     ## -------------    RUN FINISHED    ------------- ##\
        \n     ####################################################\n"

# comment out if not wanted
#poweroff