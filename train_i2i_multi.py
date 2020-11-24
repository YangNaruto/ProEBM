import torch as t
import torch.nn as nn
import torchvision as tv, torchvision.transforms as tr
import torch.nn.functional as F
import torch
import random
import argparse
import shutil
import os
from torch.utils.data import DataLoader
import math
import numpy as np
from torchvision.transforms import functional as trans_fn
from glob import glob
import cv2
import sys
import copy
from PIL import Image
sys.path.append("./utils/score")
from scipy.stats import truncnorm
from module import networks
from utils import dataset_util
from utils.logger import Logger
from utils.submit import _get_next_run_id_local, _create_run_dir_local, _copy_dir, _save_args
from utils.score import fid
from utils.util import EMA, read_image_folder, count_parameters

seed = 1
t.manual_seed(seed)
if t.cuda.is_available():
	t.cuda.manual_seed_all(seed)

sqrt = lambda x: int(t.sqrt(t.Tensor([x])))

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ProgressiveEBM(object):
	def __init__(self, args):
		super().__init__()
		self.default_lr = args.lr_default
		self.device = t.device(args.device if torch.cuda.is_available() else 'cpu')

		self.ebmA = networks.EBM(base_channel=args.base_channel, spectral=args.spectral, res=args.res,
		                                  projection=args.projection, activation_fn=args.activation_fn,
		                                  from_rgb_activate=args.from_rgb_activate, bn=args.bn, split=args.split_bn,
		                                  add_attention=args.attention).to(self.device)
		self.optimizerA = torch.optim.Adam(self.ebmA.parameters(), lr=self.default_lr, betas=(0.5, 0.999), amsgrad=args.amsgrad)

		self.ebmB = networks.EBM(base_channel=args.base_channel, spectral=args.spectral, res=args.res,
		                                  projection=args.projection, activation_fn=args.activation_fn,
		                                  from_rgb_activate=args.from_rgb_activate, bn=args.bn, split=args.split_bn,
		                                  add_attention=args.attention).to(self.device)
		self.optimizerB = torch.optim.Adam(self.ebmB.parameters(), lr=self.default_lr, betas=(0.5, 0.999), amsgrad=args.amsgrad)
		print(count_parameters(self.ebmA))
		self.args = args
		self.truncation = args.truncation
		self.min_step, self.max_step = int(math.log2(args.init_size)) - 2, int(math.log2(args.max_size)) - 2
		self.val_clip = args.val_clip
		self.noise_ratio = args.noise_ratio
		self.resolution = [2 ** x for x in range(2, 10)]
		self.progressive = args.pro

	def requires_grad(self, flag=True):
		for p in self.ebmA.parameters():
			p.requires_grad = flag

		for p in self.ebmB.parameters():
			p.requires_grad = flag

	# @torch.no_grad()
	def _momentum_update_model(self, model, model_ema, momentum=0.99):
		"""
		Momentum update of the key encoder
		"""
		for param, param_ema in zip(model.parameters(), model_ema.parameters()):
			param_ema.data = param_ema.data * momentum + param.data * (1. - momentum)

	@staticmethod
	def slerp(val, low, high):
		low_norm = low / torch.norm(low, dim=1, keepdim=True)
		high_norm = high / torch.norm(high, dim=1, keepdim=True)
		omega = torch.acos((low_norm * high_norm).sum(1))
		so = torch.sin(omega)
		res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
		return res

	@staticmethod
	def add_noise(x, sigma=1e-2):
		return  x + sigma * t.randn_like(x)

	def sample_data(self, datasetA, datasetB, batch_size, image_size=4):
		datasetA.resolution = image_size
		datasetB.resolution = image_size
		loaderA = DataLoader(datasetA, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True, pin_memory=True)
		loaderB = DataLoader(datasetB, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True, pin_memory=True)
		return loaderA, loaderB

	@staticmethod
	def truncated_normal(size, threshold=1.):
		values = truncnorm.rvs(-threshold, threshold, size=size)
		return values

	def cyclic(self, T, i, lr, M=4, min_lr=0.):
		rcounter = T + i
		cos_inner = np.pi * (rcounter % (T // M))
		cos_inner /= T // M
		cos_out = np.cos(cos_inner) + 1
		lr = float(np.clip(0.5 * cos_out * lr, min_lr, 100))
		return lr

	def polynomial(self, t, T, base_lr, end_lr=0.0001, power=1.):
		lr = (base_lr - end_lr) * ((1 - t / T) ** power) + end_lr

		# lr = a * (b + t) ** power
		return lr

	def adjust_lr(self, lr):
		if 'ab' in self.args.direction.split('_'):
			for group in self.optimizerA.param_groups:
				mult = group.get('mult', 1)
				group['lr'] = lr * mult
		if 'ba' in self.args.direction.split('_'):
			for group in self.optimizerB.param_groups:
				mult = group.get('mult', 1)
				group['lr'] = lr * mult

	def pixel_norm(self, input):
		return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

	def langevin_sampler(self, model, x_k, sampler_step, step, alpha, from_beginning=False, soft=False, return_seq=False):
		alphas = [1 for _ in range(step)]
		alphas.append(alpha)
		langevin_steps = sampler_step
		if step == self.min_step or not self.progressive:
			step_list = range(step, step + 1)
		else:
			if from_beginning:
				step_list = range(self.min_step, step + 1)
				self.noise_ratio = 0.
				x_k = F.interpolate(x_k, scale_factor= 1 / (step - self.min_step + 1), mode='nearest')
			else:
				if ((alpha < 1 or self.noise_ratio < 1) and soft):
					step_list = range(step - 1, step + 1)
					x_k = F.interpolate(x_k, scale_factor=0.5, mode='nearest')
					langevin_steps = int(sampler_step * (1 - alpha**0.5))
					# langevin_steps = int(sampler_step * (1 - alpha**0.5))
				else:
					step_list = range(step, step + 1)

		step_list = list(step_list)

		for index, s in enumerate(step_list):

			if index != 0:
				x_k = F.interpolate(x_k.clone().detach(), scale_factor=2, mode='nearest')
				# x_k = x_k + self.noise_ratio * self.sample_p_0(x_k.shape[0], x_k.shape[-1], n_ch=3, initial='gaussian')
				langevin_steps = sampler_step
				x_k = torch.clamp(x_k, -self.val_clip, self.val_clip)

			x_k.requires_grad_(True)
			sample_array = torch.zeros((x_k.shape[0], langevin_steps, x_k.shape[1], x_k.shape[2], x_k.shape[3]))
			for k in range(langevin_steps):

				loss = model(x_k, step=s, alpha=alphas[s]).sum()

				f_prime = t.autograd.grad(loss, [x_k], retain_graph=True)[0]
				# lr = self.cyclic(langevin_steps, k, M=1, lr=self.args.langevin_lr, min_lr=self.args.langevin_lr) if self.args.cyclic else self.args.langevin_lr
				# sigma = self.cyclic(langevin_steps, k, M=1, lr=1e-1, min_lr=1e-2) if self.args.cyclic else 1e-2

				lr = self.polynomial(k, langevin_steps, base_lr=self.args.langevin_lr, end_lr=self.args.langevin_lr, power=1) if self.args.cyclic else self.args.langevin_lr
				sigma = self.polynomial(k, langevin_steps, base_lr=5e-2, end_lr=1e-3, power=1) if self.args.cyclic else 0e-3

				x_k.data += lr * f_prime + sigma * t.randn_like(x_k)
				x_k.data.clamp_(-self.val_clip, self.val_clip)
				sample_array[:, k] = x_k.data

		if not return_seq:
			return x_k.clone().detach()
		else:
			return x_k.clone().detach(), sample_array

	def sample_p_d(self, img, sigma=1e-2):
		return self.add_noise(img, sigma).detach()

	def sample_p_0(self, bs, im_sz, n_ch=3, initial='gaussian'):
		if initial == 'gaussian':
			x_k = torch.from_numpy(self.truncated_normal(size=(bs, n_ch, im_sz, im_sz), threshold=self.truncation).astype(np.float32)).to(self.device)
		else:
			x_k = self.truncation*(-2. * torch.rand(bs, n_ch, im_sz, im_sz) + 1.).to(self.device)

		return x_k

	@staticmethod
	def plot(p, x):
		m = x.shape[0]
		row = int(sqrt(m))
		tv.utils.save_image(x[:row ** 2], p, padding=1, normalize=True, range=(-1., 1.), nrow=row)
		return

	@staticmethod
	def check_folder(log_dir):
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		return log_dir


	def test_image_folder(self, dataset_name, sampler_step, im_size, step, alpha, soft=False, i=0):
		test_A_files = list(glob('./dataset/{}/*.*'.format(dataset_name + '/testA')))
		test_B_files = list(glob('./dataset/{}/*.*'.format(dataset_name + '/testB')))

		root = os.path.join(self.args.run_dir, '{:06d}'.format(i))
		self.check_folder(root)
		sampler_step = 64
		batch_size = 36
		if im_size > 64:
			batch_size = 16
		N = min(len(test_A_files), len(test_B_files))
		if batch_size > N:
			print(('Warning: batch size is bigger than the data size. '
			       'Setting batch size to data size'))
			batch_size = N

		interval = 4

		#   A -> B
		if 'ab' in self.args.direction.split('_'):
			sample_image = read_image_folder(test_A_files, batch_size=batch_size, im_size=im_size, resize=True).to(self.device)
			tv.utils.save_image(sample_image, os.path.join(root, 'A.png'), padding=1, normalize=True, range=(-1., 1.), nrow=int(sqrt(batch_size)))

			image_path = os.path.join(root, 'A2B.png')
			numpy_path = os.path.join(root, 'A2B.npy')
			seq_path = os.path.join(root, 'A2B_seq.png')

			img_t, A2B_seq = self.langevin_sampler(model=self.ebmA, x_k=sample_image, sampler_step=sampler_step, step=step, alpha=alpha, soft=soft,
			                                       from_beginning=self.args.from_beginning, return_seq=True)

			tv.utils.save_image(img_t, image_path, padding=1, normalize=True, range=(-1., 1.), nrow= int(sqrt(batch_size)))
			seq_len = A2B_seq[:, ::interval].shape[1]
			tv.utils.save_image(A2B_seq[:, ::interval].view(batch_size*seq_len, 3, im_size, im_size), seq_path, padding=0, normalize=True, range=(-1., 1.),
			                    nrow = seq_len)
			# np.save(numpy_path, np.asarray([A2B_seq[i].cpu().numpy() for i in range(len(A2B_seq))]))

		if 'ba' in self.args.direction.split('_'):
			#   B -> A
			sample_image = read_image_folder(test_B_files, batch_size=batch_size, im_size=im_size, resize=True).to(self.device)
			tv.utils.save_image(sample_image, os.path.join(root, 'B.png'), padding=1, normalize=True, range=(-1., 1.), nrow=int(sqrt(batch_size)))

			image_path = os.path.join(root, 'B2A.png')
			numpy_path = os.path.join(root, 'B2A.npy')
			seq_path = os.path.join(root, 'B2A_seq.png')
			img_t, B2A_seq = self.langevin_sampler(model=self.ebmB, x_k=sample_image, from_beginning=self.args.from_beginning,
			                                       sampler_step=sampler_step, step=step, alpha=alpha, soft=soft, return_seq=True)
			# np.save(numpy_path, np.asarray([B2A_seq[i].cpu().numpy() for i in range(len(B2A_seq))]))
			tv.utils.save_image(img_t, image_path, padding=1, normalize=True, range=(-1., 1.), nrow=int(sqrt(batch_size)))
			seq_len = B2A_seq[:, ::interval].shape[1]
			tv.utils.save_image(B2A_seq[:, ::interval].view(batch_size*seq_len, 3, im_size, im_size), seq_path, padding=0, normalize=True, range=(-1., 1.),
			                    nrow= seq_len)

	def load_ckpt(self, ckpt_name):
		ckpt = torch.load(ckpt_name)
		self.ebmA.load_state_dict(ckpt['ebmA'])
		self.ebmB.load_state_dict(ckpt['ebmB'])
		self.optimizerA.load_state_dict(ckpt['optimizerA'])
		self.optimizerB.load_state_dict(ckpt['optimizerB'])
		step = ckpt['step']
		alpha = ckpt['alpha']
		used_sample = ckpt['used_sample']
		return step, alpha, used_sample

	@staticmethod
	def normalize_tensor(tensor):
		return tensor.add_(-tensor.min()).div_(tensor.max() - tensor.min() + 1e-5)

	def calculate_fid(self, stats_path, sampler_step, im_size, step, alpha, soft=False, batch_size=64, dims=2048):
		samples = self.prepare_sample(sampler_step=sampler_step, im_size=im_size, step=step, alpha=alpha, batch_size=batch_size, soft=soft)
		fid_score = fid.calculate_fid_given_sample_path(samples, stats_path, batch_size, cuda=self.device, dims=dims)
		del samples
		return fid_score

	def train(self, args, datasetA, datasetB, save_path, load=False, ckpt_name=None):
		step = int(math.log2(args.init_size)) - 2

		if load and ckpt_name is not None:
			step, alpha, used_sample = self.load_ckpt(ckpt_name)
		else:
			used_sample = 0

		base_sigma = 1e-2
		resolution = 4 * 2 ** step
		loaderA, loaderB = self.sample_data(
			datasetA, datasetB, args.batch.get(resolution, args.batch_default), resolution
		)
		data_loaderA = iter(loaderA)
		data_loaderB = iter(loaderB)
		self.adjust_lr(args.lr.get(resolution, self.default_lr))
		prev_step = 0
		max_step = int(math.log2(args.max_size)) - 2
		final_progress = False

		total_used_samples = 0
		interval = 2000
		T = args.iterations
		for i in range(T):
			self.ebmA.zero_grad()
			self.ebmB.zero_grad()
			alpha = min(1, 1 / args.phase * (used_sample + 1))

			if (resolution == args.init_size and args.ckpt is None) or final_progress:
				alpha = 1

			if used_sample > 2 * args.phase:
				torch.save(
					{
						'used_sample': used_sample,
						'alpha': alpha,
						'step': step,
						'ebmA': self.ebmA.state_dict(),
						'ebmB': self.ebmB.state_dict(),
						'optimizerA': self.optimizerA.state_dict(),
						'optimizerB': self.optimizerB.state_dict(),
					},
					f'{save_path}/train_step-{i}.model',
				)
				used_sample = 0
				step += 1

				if step > max_step:
					step = max_step
					final_progress = True
				else:
					alpha = 0

				resolution = 4 * 2 ** step
				if resolution > 32:
					interval = 4000
				loaderA, loaderB = self.sample_data(
					datasetA, datasetB, args.batch.get(resolution, args.batch_default), resolution
				)
				data_loaderA = iter(loaderA)
				data_loaderB = iter(loaderB)
				self.adjust_lr(args.lr.get(resolution, self.default_lr))
			try:
				real_imageA, _ = next(data_loaderA)
				real_imageB, _ = next(data_loaderB)

			except (OSError, StopIteration):
				data_loaderA = iter(loaderA)
				data_loaderB = iter(loaderB)
				real_imageA, _ = next(data_loaderA)
				real_imageB, _ = next(data_loaderB)

			# Sample images
			used_sample += real_imageA.shape[0]
			total_used_samples += real_imageA.shape[0]
			real_imageA, real_imageB = real_imageA.to(self.device), real_imageB.to(self.device)
			sigma = self.cyclic(T, i, base_sigma, M=1, min_lr=0.01) if args.cyclic else base_sigma

			x_p_A = self.sample_p_d(real_imageA, sigma)
			x_p_B = self.sample_p_d(real_imageB, sigma)

			if prev_step != step:
				self.plot(os.path.join(save_path, 'resolution_A_{}.png'.format(resolution)), x_p_A)
				self.plot(os.path.join(save_path, 'resolution_B_{}.png'.format(resolution)), x_p_B)
				prev_step += 1

			# % Langevin sampling
			self.ebmA.eval()
			self.ebmB.eval()
			with torch.enable_grad():
				if 'ab' in args.direction.split('_'):
					x_t_B = self.langevin_sampler(model=self.ebmA, x_k=x_p_A.clone().detach(), sampler_step=args.langevin_step, step=step, alpha=alpha,
					                              from_beginning=args.from_beginning,
					                             soft=args.soft)
				if 'ba' in args.direction.split('_'):
					x_t_A = self.langevin_sampler(model=self.ebmB, x_k=x_p_B.clone().detach(), sampler_step=args.langevin_step, step=step, alpha=alpha,
					                              from_beginning=args.from_beginning,
					                             soft=args.soft)

				# if args.c_loss:
				# 	gamma = 1
				# 	x_t_B_cyc = self.langevin_sampler(model=self.ebmA, x_k=x_t_A.clone().detach(), sampler_step=args.langevin_step, step=step, alpha=alpha,
				# 	                              from_beginning=args.from_beginning,
				# 	                              soft=args.soft)
				#
				# 	x_t_A_cyc = self.langevin_sampler(model=self.ebmB, x_k=x_t_B.clone().detach(), sampler_step=args.langevin_step, step=step, alpha=alpha,
				# 	                              from_beginning=args.from_beginning,
				# 	                              soft=args.soft)
				# else:
				# 	gamma = 0.

			self.ebmA.train()
			self.ebmB.train()

			if 'ba' in args.direction.split('_'):
				x_A = torch.cat((x_p_A, x_t_A), dim=0)
				energyB = self.ebmB(x_A, step=step, alpha=alpha)
				pos_energyB, neg_energyB = [e for e in torch.split(energyB, [x_p_A.shape[0], x_t_A.shape[0]])]
				L_B = (pos_energyB - neg_energyB).mean()
				self.optimizerB.zero_grad()
				(-L_B).backward()
				self.optimizerB.step()

			if 'ab' in args.direction.split('_'):
				x_B = torch.cat((x_p_B, x_t_B), dim=0)
				energyA = self.ebmA(x_B, step=step, alpha=alpha)
				pos_energyA, neg_energyA = [e for e in torch.split(energyA, [x_p_B.shape[0], x_t_B.shape[0]])]
				L_A = (pos_energyA - neg_energyA).mean()
				(-L_A).backward()
				self.optimizerA.step()
				self.optimizerA.zero_grad()

			if i % 100 == 0:
				if 'ba' in args.direction.split('_'):
					print('Itr: {:>6d}, Kimg: {:>8d}, Alpha: {:>3.2f}, Res: {:>3d}, f(ba_p)={:8.4f}, f(ba_n)={:>8.4f}'.
					      format(i, total_used_samples // 1000, alpha, resolution, pos_energyB.mean(), neg_energyB.mean()))
					self.plot(os.path.join(save_path, '{:>06d}_t_A.png'.format(i)), x_t_A)

				if 'ab' in args.direction.split('_'):
					print('Itr: {:>6d}, Kimg: {:>8d}, Alpha: {:>3.2f}, Res: {:>3d}, f(ab_p)={:8.4f}, f(ab_n)={:>8.4f}'.
					      format(i, total_used_samples // 1000, alpha, resolution, pos_energyA.mean(), neg_energyA.mean()))
					self.plot(os.path.join(save_path, '{:>06d}_t_B.png'.format(i)), x_t_B)
				if i % 200 == 0:
					self.test_image_folder(dataset_name=args.dataset, sampler_step=args.langevin_step, im_size=self.resolution[step], step=step, alpha=alpha, soft=args.soft, i=i)

				# self.plot(os.path.join(save_path, '{:>06d}_p_A.png'.format(i)), x_p_A)
				# self.plot(os.path.join(save_path, '{:>06d}_p_B.png'.format(i)), x_p_B)

				if i % interval == 0 and i != 0 and alpha==1:
					torch.save(
						{
							'used_sample': used_sample,
							'alpha': alpha,
							'step': step,
							'ebmA': self.ebmA.state_dict(),
							'ebmB': self.ebmB.state_dict(),
							'optimizerA': self.optimizerA.state_dict(),
							'optimizerB': self.optimizerB.state_dict(),
						},
						f'{save_path}/train_step-{i}.model',
					)
		return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
	# Set dataset
	parser.add_argument('--dataset', type=str, default='cat2dog', help='cifar10/ffhq-128/celeba')
	parser.add_argument('--data_root', type=str, default="./dataset/", help='path of specified dataset')

	# Training stuff
	parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
	parser.add_argument('--device', type=str, default='0')

	parser.add_argument('--direction', type=str, default='ab', help='which direction to translate')
	parser.add_argument('--mode', type=str, default='train', help="train/eval")
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
	parser.add_argument('--sched', action='store_true', help='use lr scheduling')
	parser.add_argument('--init_size', default=8, type=int, help='initial image size')
	parser.add_argument('--max_size', default=32, type=int, help='max image size')
	parser.add_argument('--val_clip', default=1.1, type=float, help='x clip')
	parser.add_argument('--noise_ratio', default=1.0, type=float, help='noise amount in transition')
	parser.add_argument('--base_sigma', default=0.02, type=float, help='noise amount to real data')
	parser.add_argument('--conditional', action='store_true', default=False, help="add spectral normalization")
	parser.add_argument('--res', action='store_true', default=False)
	parser.add_argument('--projection', action='store_true', default=False)
	parser.add_argument('--from_beginning', action='store_true', default=False)
	parser.add_argument('--soft', action='store_true', default=False, help="if add soft connection in transition stage")
	parser.add_argument('--split_bn', action='store_true', default=False, help="if split bn")
	parser.add_argument('--ema', action='store_true', default=False)
	parser.add_argument('--initial', default='uniform', type=str, help='initial distribution')
	parser.add_argument('--langevin_step', default=50, type=int, help='langevin update steps')
	parser.add_argument('--momentum', default=0.9, type=float, help='ema momentum')
	parser.add_argument('--ema_start', default=0, type=int, help='ema start')
	parser.add_argument('--langevin_lr', default=2., type=float, help='langevin learning rate')
	parser.add_argument('--l2_coefficient', default=0.0, type=float, help='l2 regularization coefficient')
	parser.add_argument('--iterations', default=1_000_000, type=int, help='l2 regularization coefficient')
	parser.add_argument('--pro', action='store_true', default=False, help='if progressivly fading')
	parser.add_argument('--spectral', action='store_true', default=False, help="add spectral normalization")
	parser.add_argument('--bn', action='store_true', default=False, help="add split batch normalization")
	parser.add_argument('--attention', action='store_true', default=False, help='if add attention')
	parser.add_argument('--activation_fn', type=str, default='lrelu', help='activation function')
	parser.add_argument('--cyclic', action='store_true', default=False, help='if apply cyclic langevin learning rate')
	parser.add_argument('--amsgrad', action='store_true', default=False, help='if apply amsgrad')
	parser.add_argument('--c_loss', action='store_true', default=False, help='if apply amsgrad')

	# save stuff
	parser.add_argument('--name_suffix', default='', type=str, help='save path')
	parser.add_argument('--load', action='store_true', default=False, help='if progressivly fading')
	parser.add_argument('--eval_step', default=50, type=int, help='evaluate step')
	parser.add_argument('--run_dir', default='', type=str, help='running directory')
	parser.add_argument('--stats_path', default='./stats/fid_stats_cifar10_train.npz', type=str, help='running directory')
	parser.add_argument('--truncation', default=1.0, type=float, help="truncation threshold")

	parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
	parser.add_argument('--base_channel', default=32, type=int, help='the base number of filters')
	parser.add_argument(
		'--from_rgb_activate',
		action='store_true', default=False,
		help='use activate in from_rgb (original implementation)',
	)
	parser.add_argument('--save_path', default='results', type=str)
	args = parser.parse_args()

	# assert (args.pro is False and args.init_size==args.max_size) or args.pro is True, "the init_size must be equal to max_size when not to use fading"
	# args.device = torch.device('cuda:{}'.format(args.device))
	args.device = torch.cuda.current_device()
	save_path = os.path.join(args.save_path, args.dataset)
	save_path = save_path + '_' + args.name_suffix if args.name_suffix != '' else save_path
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	if args.mode == "train":
		dir_name = f'step{args.langevin_step}-trunc{args.truncation}-res{args.max_size}-lr{args.langevin_lr}-{args.initial}-ch{args.base_channel}-{args.activation_fn}-' \
		           f'{args.direction}'
		if args.ema:
			dir_name += f'-ema{args.momentum}'
		run_dir = _create_run_dir_local(save_path, dir_name)
		_copy_dir(['module', 'utils', 'train_i2i.py'], run_dir)
		sys.stdout = Logger(os.path.join(run_dir, 'log.txt'))

		args.run_dir = run_dir


		dataset_path_A = os.path.join(args.data_root, os.path.join(args.dataset, 'trainA_lmdb'))
		dataset_path_B = os.path.join(args.data_root, os.path.join(args.dataset, 'trainB_lmdb'))
		dataset_A = dataset_util.MultiResolutionDatasetTranslate(dataset_path_A)
		dataset_B = dataset_util.MultiResolutionDatasetTranslate(dataset_path_B)


		if args.sched:
			args.lr = {8: 0.0005, 16: 0.0008, 32: 0.0010, 64: 0.0012, 128: 0.0015, 256: 0.002}
			args.batch = {8: 128, 16: 128, 32: 64, 64: 64, 128: 32, 256: 16}

			args.lr = {key: val + 5e-4 for key, val in args.lr.items()}
		else:
			args.lr = {}
			args.batch = {}
		args.batch_default = 64
		args.lr_default = 5e-4
		print(args)
		trainer = ProgressiveEBM(args)

		trainer.train(args, dataset_A, dataset_B, run_dir, load=args.load, ckpt_name=args.ckpt)

	elif args.mode == "eval" and args.ckpt is not None:
		trainer = ProgressiveEBM(args)
		trainer.evaluate(ckpt_name=args.ckpt, sampler_step=args.eval_step)
