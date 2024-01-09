#!/bin/bash     
#SBATCH -p kshdtest
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH -J opi.test
#SBATCH -o ./result.log
#SBATCH -e ./output.%j.e

module switch compiler/rocm/4.0.1

#export PATH1 = 'example1'
#echo $PATH1
python gpu_cost.py ../example1/
python calculate.py example1

