#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python train_v0.py --phase 400000 --sched \
	--dataset celeba --name_suffix activation --activation_fn lrelu \
	--init_size 32 --max_size 32 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 60 --langevin_lr 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/celeba.npz \
