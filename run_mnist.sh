#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python train_v0.py --phase 150000 --sched --spectral --res --pro --from_beginning \
	--dataset mnist --name_suffix run --activation_fn swish --base_sigma 0.03 \
	--init_size 8 --max_size 32 --initial uniform \
	--noise_ratio 1.0 --base_channel 16 \
	--langevin_step 50 --langevin_lr 1 --val_clip 1.0 \
	--truncation 1.0 \
