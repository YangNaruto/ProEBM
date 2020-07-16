#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_v0.py --phase 300000 --spectral --res --soft \
	--dataset metfaces --name_suffix test --activation_fn gelu --base_sigma 0.03 \
	--init_size 8 --max_size 64 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 75 --langevin_lr 5.0 --val_clip 1.2 \
	--truncation 1.0 \
	--stats_path ./stats/metfaces64.npz \
