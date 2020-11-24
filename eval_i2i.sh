#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
python train_i2i.py --mode eval --ckpt /home/yang/ProEBM/results/vangogh2photo/00002-step50-trunc1.0-res256-lr2.0-uniform-ch16-gelu-ba/train_step-20000.model \
	 --phase 200000 --spectral --res --pro --direction ba \
	--dataset vangogh2photo --sched \
       	--activation_fn gelu --base_sigma 0.03 \
	--init_size 256 --max_size 256 --initial uniform \
	--noise_ratio 1.0 --base_channel 16 \
	--langevin_step 50 --langevin_lr 2.0 --val_clip 1.0 \
	--truncation 1.0 \
