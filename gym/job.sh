#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --partition=es1
#SBATCH --gres=gpu:1
#SBATCH --account=pc_dl4acc
#SBATCH --mail-type=all
#SBATCH --mail-user=qdu@lbl.gov
#SBATCH --qos=es_normal
#SBATCH --time=1:00:00
source ~/.venv/dqn/bin/activate
python rl_train_sb3.py -v -t 0.7
