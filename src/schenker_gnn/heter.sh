#!/bin/bash
##SBATCH --mem=128G
#SBATCH --job-name=grad_accm_sck
#SBATCH -t 48:00:00  # time requested in hour:minute:second
#SBATCH --gres=gpu:2
#SBATCH --partition=compsci-gpu
#SBATCH --output=homosout/%x-%j.out  # Save stdout to sout directory
#SBATCH --error=homosout/%x-%j.err   # Save stderr to sout directory


# python dataloader.py
# python skeleton.py
# first try using the original cluster to predict
# python key_pred.py
# python train.py
# python naive_train.py
# python grad_accm.py
python train.py
# python diffpool.py
# python test_coor.py
# then try using the skn analysis to predict
