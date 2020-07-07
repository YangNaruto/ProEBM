#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python train_v0.py --phase 200000 --sched \
	--dataset metfaces --name_suffix downsample --activation_fn lrelu \
	--init_size 8 --max_size 64 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 60 --langevin_lr 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/metfaces64.npz \
