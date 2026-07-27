#!/usr/bin/env bash
#SBATCH --no-requeue
#SBATCH --mem 6G
#SBATCH -p genoa64
#SBATCH --qos pipelines
#SBATCH --output=
#SBATCH --error=
#SBATCH --job-name nf-genopre

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
module load Java/21.0.2
module load EasyBuild/4.9.4

# limit the RAM that can be used by nextflow
export NXF_JVM_ARGS="-Xms2g -Xmx5g"
export NXF_VER=24.10.1

# Run the pipeline. The command uses the arguments passed to this script, e.g:
#
# $ sbatch submit_nf.sh nextflow/rnatoy -with-singularity
#
# will use "nextflow/rnatoy -with-singularity" as arguments
#nextflow run -ansi-log false "$@" & pid=$!
#not necessary

mkdir -p ./log/
mkdir -p ./trace/ 

mydate=$(date +"%Y%m%d_%H%M")

# -qs is for queue size
nextflow run main.nf \
    -ansi-log false \
    -w nf_workdir/genotype_prep/work \
    -params-file params_list.yaml \
    -profile crg -with-trace -resume > log/genotypes_${mydate}.log & pid=$!
    
# Wait for the pipeline to finish
echo "Waiting for ${pid}"
wait $pid

mv ./trace-* ./trace/

# Return 0 exit-status if everything went well
exit 0

#sbatch sbatch_pipeline.sh
