# How to run this

## Step 1: Train and test ML model

```bash
./ml.sh ML_WORKDIR
```

## Step 2: Create friend datasets with ML application

```bash
# Create jobs with model application
ml/create_jobs.sh ML_WORKDIR
cd ML_WORKDIR/MLScores_jobs
condor_submit job.jdl

# Check model application and eventually run remaining jobs locally
ml/check.sh ML_WORKDIR

# Merge application files
ml/merge.sh ML_WORKDIR
```

## Step 3: Produce histograms for the analysis

```bash
# Create jobs with shape production
# NOTE: Point to the ML shapes in utils/config.py
shapes/create_jobs.sh SHAPES_WORKDIR
cd SHAPES_WORKDIR/shapes_jobs
condor_submit job.jdl

# Check shape production and eventually run jobs locally
shapes/check.sh SHAPES_WORKDIR

# Merge histograms
shapes/merge.sh SHAPES_WORKDIR
```

# Step 4: Run postprocessing

```bash
./postproc.sh SHAPES_WORKDIR
````
