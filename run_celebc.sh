#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_v0.py --phase 400000 --sched --spectral --res --cyclic --soft \
	--dataset celeba-c --name_suffix 64 --activation_fn swish --base_sigma 0.02 \
	--init_size 8 --max_size 64 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 75 --langevin_lr 1.0 --val_clip 1.0 \
	--truncation 1.0 \
	--stats_path stats/celebac64.npz
