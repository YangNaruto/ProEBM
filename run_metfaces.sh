#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python train_v0.py --phase 100000 --spectral --ema --momentum 0.9 \
	--dataset metfaces --name_suffix run --activation_fn gelu --base_sigma 0.05 \
	--init_size 8 --max_size 64 --initial gaussian \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 50 --langevin_lr 2.0 \
	--truncation 1.0 \
	--stats_path ./stats/metfaces64.npz \
