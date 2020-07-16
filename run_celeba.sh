#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_v0.py --phase 1500000 --sched --spectral --soft --res \
	--dataset celeba --name_suffix run --activation_fn gelu --base_sigma 0.02 \
	--init_size 8 --max_size 64 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 60 --langevin_lr 10.0 --val_clip 1.1 \
	--truncation 1.0 \
	--stats_path ./stats/celeba64.npz \
