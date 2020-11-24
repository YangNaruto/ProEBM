#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_v1.py --phase 4000000 --sched --spectral --soft --cyclic --res --from_beginning \
	--dataset imagenet32x32 --name_suffix run --activation_fn swish --base_sigma 0.00 \
	--init_size 8 --max_size 32 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 15 --langevin_lr 1.0 --val_clip 1.0 \
	--truncation 1.0 \
	--stats_path ./stats/imagenet32x32.npz \
