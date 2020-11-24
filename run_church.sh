#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python train_v1.py --phase 1000000 --cyclic --sched --spectral --soft --res --from_beginning \
	--dataset church --name_suffix fix --activation_fn swish --base_sigma 0.00 \
	--init_size 8 --max_size 128 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 15 --langevin_lr 1.0 --val_clip 1.0 \
	--truncation 1.0 \
