#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train_v1.py --phase 1200000 --sched --spectral --res --cyclic --soft --from_beginning \
	--dataset celeba-c --name_suffix fix --activation_fn swish --base_sigma 0.00 \
	--init_size 8 --max_size 64 --initial uniform \
	--pro --noise_ratio 1.0 --base_channel 32 \
	--langevin_step 20 --langevin_lr 1.0 --val_clip 1.0 \
	--truncation 1.0 \
	--stats_path stats/celebac64.npz
