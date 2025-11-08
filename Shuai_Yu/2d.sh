#!/bin/bash -l

#SBATCH --job-name=2DCNSLDC
#SBATCH --time=03:59:59
#SBATCH --nodes=1
#SBATCH --nodelist=adroit-h11n5

Ma="0.025"
N="64"
dt="0.001"
t_final="500.0"
sampleinterval=5000

echo "Starting job..."
srun ./solver $Ma $N $dt $t_final $sample_interval
echo "Job finished..."
