#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --partition=v100_sxm2_4
#SBATCH --time=96:00:00
#SBATCH --job-name=TimeFaces
#SBATCH --mail-type=END
#SBATCH --mail-user=alf590@nyu.edu
#SBATCH --output=TimeFaces_%j.out

module purge

module load cuda/10.0.130
module load cudnn/10.0v7.4.2.24
module load python3/intel/3.6.3

source /home/alf590/venv/bin/activate

pip install --upgrade tensorflow-gpu==1.14
cd /home/alf590/stylegan2-ada/
pip install image
pip install numpy==1.16.4
python train.py --resume=ffhq1024 --outdir=/scratch/alf590/ --gpus=4 --data=/scratch/alf590/datasets/1024timeFaces --metrics=none --cfg=aidan
