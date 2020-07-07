#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python train_v0.py --phase 500000 --sched --res --projection --spectral \
	--dataset cifar10 --name_suffix downsample --activation_fn gelu \
	--init_size 8 --max_size 32 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 60 --langevin_lr 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/cifar10.npz \
