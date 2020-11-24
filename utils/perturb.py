from glob import glob
from util import EMA, read_image_folder, count_parameters, read_single_image, plot_heatmap
import torch
import torchvision as tv
import os

sigmas= [0, 0.01, 0.02, 0.03]
root = "/home/yzhao/datasets/lsun/church_train/church_train"
files = list(glob(os.path.join(root, '*.jpg')))

for sigma in sigmas:
	for i, file in enumerate(files[:5]):
		img = read_single_image(file, im_size=128, resize=True)
		img_ = img + sigma * torch.randn_like(img)
		tv.utils.save_image(img_, 'perturb_%d_%f.png'%(i, sigma), padding=0, normalize=True, range=(-1., 1.), nrow=1)




