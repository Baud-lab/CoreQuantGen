#!/usr/bin/env bash
#SBATCH --no-requeue
#SBATCH --mem 6G
#SBATCH -p genoa64
#SBATCH --qos pipelines
#SBATCH --output=/nfs/scratch01/abaud/htonnele/logs/nf-pipelines/%x_%A.out
#SBATCH --error=/nfs/scratch01/abaud/htonnele/logs/nf-pipelines/%x_%A.err
#SBATCH --job-name VD_HSmice

# Configure bash
set -e          # exit immediately on error
set -u          # exit immidiately if using undefined variables
set -o pipefail # ensure bash pipelines return non-zero status if any of their command fails

# Setup trap function to be run when canceling the pipeline job. It will propagate the SIGTERM signal
# to Nextlflow so that all jobs launche by the pipeline will be cancelled too.
_term() {
  echo "Caught SIGTERM signal!"
  kill -s SIGTERM $pid
  wait $pid
}

trap _term TERM

# load Java module
module load Java
module load Python/3.10.4-GCCcore-11.3.0
module load Nextflow/25.04.6

# limit the RAM that can be used by nextflow
export NXF_JVM_ARGS="-Xms2g -Xmx5g"

# Making directories if needed
echo "making output directory if needed"

mkdir -p "./log/"
mkdir -p "./trace/"
mkdir -p "./output/HSmice/"

WORKMAIN="/nfs/scratch01/abaud/htonnele/nf_PRJs/nf-CoreQuantGen/realdata/"
mkdir -p "$WORKMAIN/work_HSmice"

PARFILE="params/params.HSmice_bi.yaml"
#PARFILE="params/params.HSmice_uni.yaml"
#PARFILE="params/params.HSmice_sex.yaml"

mydate=$(date +"%Y%m%d_%H%M")

echo "now in: "
pwd
# Now in project directory to run nextflow

# Run the pipeline. The command uses the arguments passed to this script, e.g:
#
# $ sbatch submit_nf.sh nextflow/rnatoy -with-singularity
#
# will use "nextflow/rnatoy -with-singularity" as arguments
#nextflow run -ansi-log false "$@" & pid=$!

# Running nextflow command 

echo "launching nextflow run"
nextflow run -ansi-log false main.nf \
             -profile crg \
             -params-file $PARFILE -c nextflow.config \
             -w $WORKMAIN/work_HSmice/ \
             -with-trace -resume > log/nfRun_HSmice_${mydate}.log & pid=$!

# Wait for the pipeline to finish
echo "Waiting for ${pid}"
wait $pid

# Move trace to trace folder
TRACE=$(basename trace-*)
mv -v $TRACE trace/${TRACE}.HSmice

# Return 0 exit-status if everything went well
exit 0

# Cmd to run
#sbatch submit_HSmice.sh