#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_multi.py --phase 600000 --sched --projection --spectral --soft --ema \
	--dataset celeba --name_suffix grid --activation_fn elu --base_sigma 0.02 \
	--init_size 8 --max_size 64 --initial uniform --momentum 0.9 \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 50 --langevin_lr 2.0 --val_clip 1.0 \
	--truncation 0.95 \
	--stats_path ./stats/celeba64.npz \
