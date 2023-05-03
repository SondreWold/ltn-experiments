#!/bin/bash
#SBATCH --job-name=LTN
#SBATCH --account=ec30
#SBATCH --mail-type=FAIL
#SBATCH --time=04:00:00
#SBATCH --partition=ifi_accel --gpus=rtx30:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#
# by default, request two cores (NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --ntasks-per-node=2


#
# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
#
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
#module use -a /fp/projects01/ec30/software/easybuild/modules/all/
#module load nlpl-nlptools/2022.01-foss-2021a-Python-3.9.5
#module load nlpl-gensim/4.2.0-foss-2021a-Python-3.9.5
#module load nlpl-pytorch/1.11.0-foss-2021a-cuda-11.3.1-Python-3.9.5
#module load nlpl-transformers/4.24.0-foss-2021a-Python-3.9.5
export PS1=\$
source venv/bin/activate

file=$1
# print information (optional)
echo "submission directory: ${SUBMITDIR}"
ulimit -a
module list

# by default, pass on any remaining command-line options
echo $file
chmod u+x $file
./$file
