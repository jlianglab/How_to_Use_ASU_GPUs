#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p yyanggpu1
#SBATCH -q wildfire
#SBATCH --gres=gpu:1
#SBATCH --mem 75G
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nuislam@asu.edu

module load anaconda3/5.3.0
conda activate tf-1-gpu

~/.conda/envs/tf-1-gpu/bin/python nih14.py --run 205 --batch_size 128 --backbone resnet50_imgnet --gpu 0

# ~/.conda/envs/tf-1-gpu/bin/python nih14.py --run 204 --batch_size 128 --backbone resnet50_imgnet --weight saved_models/nih14_resnet50_imgnet_run204/ckpt_epoch_49.pth --test True







# python nih14.py --run 101 --batch_size 128 --backbone resnet50 --weight saved_models/ckpt_epoch_55.pth --test True






