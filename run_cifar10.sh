#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python train_v0.py --phase 500000 --sched --res --spectral --projection --soft \
	--dataset cifar10 --name_suffix act_full --activation_fn lrelu --base_sigma 0.01 \
	--init_size 8 --max_size 32 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 50 --langevin_lr 1.0 --val_clip 1.1 \
	--truncation 1.0 \
	--stats_path ./stats/cifar10.npz \
