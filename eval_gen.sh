#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_v0.py --mode eval --phase 500000 --sched --res --spectral \
	--ckpt /home/yzhao/ProEBM/results/celeba-c_64/00001-step50-trunc1.0-res64-lr1.0-uniform-ch48-swish/train_step-216000.model\
	--dataset celeba-c --activation_fn swish \
	--init_size 8 --max_size 64 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 48 \
	--langevin_step 50 --langevin_lr 1.0 \
	--truncation 1.0