#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
python train_i2i.py --phase 100000 --spectral --pro --soft --direction ab \
	--dataset cat2dog \
       	--activation_fn elu --base_sigma 0.03 \
	--init_size 16 --max_size 64 --initial uniform \
	--noise_ratio 0.02 --base_channel 32 \
	--langevin_step 60 --langevin_lr 1.0 --val_clip 1.0 \
	--truncation 1.0 \
