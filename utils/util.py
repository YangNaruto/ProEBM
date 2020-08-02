import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as trans_fn
import torchvision.transforms as transforms
import random

class EMA(object):
	def __init__(self, source, target, decay=0.9999, start_itr=0):
		self.source = source
		self.target = target
		self.decay = decay
		# Optional parameter indicating what iteration to start the decay at
		# Initialize target's params to be source's
		self.source_dict = self.source.state_dict()
		self.target_dict = self.target.state_dict()
		print('Initializing EMA parameters to be source parameters...')
		with torch.no_grad():
			for key in self.source_dict:
				self.target_dict[key].data.copy_(self.source_dict[key].data)
				# target_dict[key].data = source_dict[key].data # Doesn't work!

	def update(self, decay=0.0):

		with torch.no_grad():
			for key in self.source_dict:
				self.target_dict[key].data.copy_(self.target_dict[key].data * decay
				                                 + self.source_dict[key].data * (1 - decay))

def imread(filename, size, resize=True):
	"""
	Loads an image file into a (height, width, 3) uint8 ndarray.
	"""
	img = Image.open(filename)
	if resize:
		img = im_resize(img, size=size)
	return np.asarray(img, dtype=np.uint8)[..., :3]

def im_resize(img, size=128):
	img = trans_fn.resize(img, size, Image.LANCZOS)
	img = trans_fn.center_crop(img, size)
	return img

def read_image_folder(files, batch_size, im_size, resize=True):
	random.shuffle(files)

	images = np.array([imread(str(f), size=im_size, resize=resize).astype(np.float32)
	                   for f in files[:batch_size]])

	# Reshape to (n_images, 3, height, width)
	images = images.transpose((0, 3, 1, 2))
	images = (images / 255 - 0.5) * 2
	batch = torch.from_numpy(images).type(torch.FloatTensor)

	return batch