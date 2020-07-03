#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_v0.py --device 0 --phase 500000 --sched --res --spectral --projection \
	--dataset celeba --name_suffix llr --activation_fn gelu \
	--init_size 16 --max_size 32 --initial uniform \
	--pro --noise_ratio 0.5 --base_channel 32 \
	--langevin_step 60 --langevin_lr 1.5 \
	--truncation 1.0 \
	--stats_path ./stats/celeba32.npz \
