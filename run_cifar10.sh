#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_v0.py --phase 500000 --sched \
	--dataset cifar10 --name_suffix init --activation_fn gelu \
	--init_size 32 --max_size 32 --initial gaussian \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 60 --langevin_lr 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/cifar10.npz \
