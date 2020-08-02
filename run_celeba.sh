#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_v0.py --phase 600000 --sched --projection --spectral --soft --ema --cyclic \
	--dataset celeba --name_suffix run --activation_fn elu --base_sigma 0.02 \
	--init_size 8 --max_size 64 --initial uniform --momentum 0.95 \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 60 --langevin_lr 2.0 --val_clip 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/celeba64.npz \
