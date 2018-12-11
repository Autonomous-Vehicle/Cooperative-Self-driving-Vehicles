#!/bin/bash                                   
#SBATCH --nodes=1				#Get one node
#SBATCH --cpus-per-task=8			#8 cores per task
#SBATCH --ntasks=1				#But only one task
#SBATCH --gres=gpu:2				#And one GPU
#SBATCH --gres-flags=enforce-binding		#Insist on good CPU/GPU alignment
#SBATCH --time=90:00:00				#Run for 10 minutes, at most
#SBATCH --job-name=CAP5415run			#Name the job so I can see it in squeue
#SBATCH --output=out/CAP5415run-train-slurm-%J.out
#SBATCH --mem=131072

#SBATCH --mail-type=BEGIN,END,FAIL		#Send me email for various states
#SBATCH --mail-user rodovr90@gmail.com		#Use this address

#Give this process 1 task (per GPU, but only one GPU), then
#assign eight 8per task (so 8 cores overall). Then enforce
#that slurm assigns only CPUs that are on the socket closest
#to the GPU you get
# Other comands
#--mem-per-cpu=16384

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=2
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the TensorFlow module
module load tensorflow/tensorflow-1.6.0
# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
nvidia-smi

echo

# Activate the GPU version of TensorFlow
source activate tensorflow-gpu

# Run TensorFlow
echo
time python /home/cap5415.student9/CAP5415/run_test.py
echo

# You’re done!
echo "Ending script..."
date
