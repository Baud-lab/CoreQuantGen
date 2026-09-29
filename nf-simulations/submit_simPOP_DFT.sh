#!/usr/bin/env bash
#SBATCH --no-requeue
#SBATCH --mem 6G
#SBATCH -p genoa64
#SBATCH --qos pipelines
#SBATCH --output=<scratch_path>/logs/nf-pipelines/%x_%A.out ## EDIT HERE
#SBATCH --error=<scratch_path>/logs/nf-pipelines/%x_%A.err ## EDIT HERE 
#SBATCH --job-name ... ## EDIT HERE: give job name

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
mkdir -p "./output/sim<POP>/" ## EDIT HERE: with name of population/dataset

# define params file
PARFILE="params/params.sim<POP>.set<SETNAME>.yaml" ## EDIT HERE: with path to params

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
              -c nextflow.config -params-file $PARFILE \
             -w <scratch_dir>/nf_workdir/CoreQuantGen/simulations/work_sim<POP>/ \
             -with-trace -resume > log/sim<POP>_set<NAME>_${mydate}.log & pid=$!
             ## EDIT HERE: -w <scratch_dir>/nf_workdir/CoreQuantGen/simulations/work_sim<POP>/ : with work directory 
             ##            log/sim<POP>_set<NAME>${mydate}.log                                 : name of population/dataset and name of set

# Wait for the pipeline to finish
echo "Waiting for ${pid}"
wait $pid

# Move trace to trace folder
mv ./trace-* ./trace/

# Return 0 exit-status if everything went well
exit 0

# Cmd to run
#sbatch submit_sim<POP>.sh ## EDIT HERE: with he name of the file