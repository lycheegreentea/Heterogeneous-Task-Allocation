#!/bin/bash
#SBATCH --job-name=run_simulator_gpu
#SBATCH --partition=csi
#SBATCH --qos=csi
#SBATCH --account=csivisitors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:15:00


module load python
source /direct/sdcc+u/lchen6/venv/bin/activate
python /direct/sdcc+u/lchen6/run_cpu_tests.py