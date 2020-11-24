#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_i2i.py --phase 200000 --spectral --pro --res --direction ab \
	--dataset portrait --sched \
       	--activation_fn swish --base_sigma 0.00 \
	--init_size 256 --max_size 256 --initial uniform \
	--noise_ratio 1.0 --base_channel 16 \
	--langevin_step 50 --langevin_lr 1.0 --val_clip 1.0 \
	--truncation 1.0 \
