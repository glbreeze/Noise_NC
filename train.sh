#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl

# job info
NOISE=$1
LOSS=$2
ETF=$3


python main.py --dset cifar10 --model resnet18 --wd 5e-4 --max_epochs 300 --lr 0.05 \
    --scheduler ms --noise_rate ${NOISE}  --loss ${LOSS}  --ETF_fc ${ETF} \
    --exp_name cf10_noise${NOISE}_ETF${ETF}_${LOSS}