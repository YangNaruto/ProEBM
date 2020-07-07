#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_v0.py --phase 400000 --sched --res \
	--dataset celeba --name_suffix activation --activation_fn gelu \
	--init_size 16 --max_size 32 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 50 --langevin_lr 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/celeba32.npz \
