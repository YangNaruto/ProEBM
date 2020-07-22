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
import sys
import copy
sys.path.append("./utils/score")
from scipy.stats import truncnorm
from module import networks
from utils import dataset_util
from utils.logger import Logger
from utils.submit import _get_next_run_id_local, _create_run_dir_local, _copy_dir, _save_args
from utils.score import fid
from utils.util import EMA
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

		num_classes = 10 if args.dataset == 'cifar10' else 1000

		self.ebm = networks.Discriminator(base_channel=args.base_channel, spectral=args.spectral, res=args.res,
		                                  projection=args.projection, activation_fn=args.activation_fn,
		                                  from_rgb_activate=args.from_rgb_activate, bn=args.bn, split=args.split_bn,
		                                  add_attention=args.attention, num_classes=num_classes).to(self.device)

		# create ema model
		self.ebm_ema = networks.Discriminator(base_channel=args.base_channel, spectral=args.spectral, res=args.res,
		                                  projection=args.projection, activation_fn=args.activation_fn,
		                                  from_rgb_activate=args.from_rgb_activate, bn=args.bn, split=args.split_bn,
		                                      add_attention=args.attention, num_classes=num_classes).to(self.device)
		self.ebm_ema.train(False)
		self.ema_manager = EMA(self.ebm, self.ebm_ema)
		# self._momentum_update_model(self.ebm, self.ebm_ema, momentum=0.0)

		self.args = args
		self.truncation = args.truncation
		self.min_step, self.max_step = int(math.log2(args.init_size)) - 2, int(math.log2(args.max_size)) - 2
		self.val_clip = args.val_clip
		self.noise_ratio = args.noise_ratio
		self.resolution = [2 ** x for x in range(2, 10)]
		self.progressive = args.pro
		self.optimizer = torch.optim.Adam(self.ebm.parameters(), lr=self.default_lr, betas=(0.0, 0.99), amsgrad=args.amsgrad)


	def requires_grad(self, flag=True):
		for p in self.ebm.parameters():
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

	def metropolis_hastings(self, x, x_prev, step, alpha):
		energy = self.ebm(x, step=step, alpha=alpha)
		energy_prev = self.ebm(x_prev, step=step, alpha=alpha)
		acceptance = (torch.exp(-energy) / torch.exp(-energy_prev) - energy.uniform_(0., 1.)) >= 0
		x = torch.where(acceptance.view(-1, 1, 1, 1), x, x_prev)
		return x.data

	@staticmethod
	def add_noise(x, sigma=1e-2):
		return  x + sigma * t.randn_like(x)

	def sample_data(self, dataset, batch_size, image_size=4):
		dataset.resolution = image_size
		loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True, pin_memory=True)
		return loader

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

	def adjust_lr(self, lr):
		for group in self.optimizer.param_groups:
			mult = group.get('mult', 1)
			group['lr'] = lr * mult

	def langevin_sampler(self, x_k, sampler_step, step, alpha, from_beginning=False, label=None, soft=False, sampling=False):
		alphas = [1 for _ in range(step)]
		alphas.append(alpha)

		if step == self.min_step or not self.progressive:
			step_list = range(step, step + 1)
		else:
			if from_beginning:
				step_list = range(self.min_step, step + 1)
				self.noise_ratio = 0.
			else:
				step_list = range(step - 1, step + 1) if (alpha < 1 or self.noise_ratio < 1 and soft) else range(step, step+1)

		# initial = self.args.initial if self.args.noise_ratio == 1.0 else 'gaussian'
		step_list = list(step_list)
		x_k.requires_grad_(True)
		beta = 0.0 if self.args.initial == 'gaussian' else 0.0
		for index, s in enumerate(step_list):
			if index != 0:
				# x_k = self.ebm.norm[s](x_k)
				x_k = F.interpolate(x_k, scale_factor=2, mode='nearest')
				x_k = (1 - self.noise_ratio * alphas[s]) * x_k + \
				      self.noise_ratio * alphas[s] * self.sample_p_0(x_k.shape[0], x_k.shape[-1], n_ch=3, initial=self.args.initial)

				x_k = torch.clamp(x_k, -self.val_clip, self.val_clip)

			# langevin_steps = int(sampler_step / len(step_list))
			# langevin_steps = int(sampler_step / (len(step_list) - index))
			langevin_steps = sampler_step
			e = 0.1
			for k in range(langevin_steps):

				if self.args.ema and sampling:
					loss = self.ebm_ema(x_k, step=s, alpha=alphas[s], label=label).sum() + beta * (-x_k**2 / (2*e**2)).sum()
				else:
					loss = self.ebm(x_k, step=s, alpha=alphas[s], label=label).sum() + beta * (-x_k**2 / (2*e**2)).sum()

				f_prime = t.autograd.grad(loss, [x_k], retain_graph=True)[0]
				lr = self.cyclic(langevin_steps, k, M=1, lr=self.args.langevin_lr, min_lr=2.0) if self.args.cyclic else self.args.langevin_lr
				sigma = self.cyclic(langevin_steps, k, M=1, lr=1e-1, min_lr=5e-3) if self.args.cyclic else 5e-3

				x_k.data += lr * f_prime + sigma * t.randn_like(x_k)
			x_k.data.clamp_(-self.val_clip, self.val_clip)

		return x_k.clone().detach()

	def sample_p_0(self, bs, im_sz, n_ch=3, initial='gaussian'):
		if initial == 'gaussian':
			x_k = torch.from_numpy(self.truncated_normal(size=(bs, n_ch, im_sz, im_sz), threshold=self.truncation).astype(np.float32)).to(self.device)
		else:
			x_k = self.truncation*(-2. * torch.rand(bs, n_ch, im_sz, im_sz) + 1.).to(self.device)

		return x_k

	def sample_p_d(self, img, sigma=1e-2):
		return self.add_noise(img, sigma).detach()

	@staticmethod
	def plot(p, x):
		m = x.shape[0]
		row = int(sqrt(m))
		tv.utils.save_image(x[:row ** 2], p, padding=1, normalize=True, range=(-1., 1.), nrow=row)
		return

	def load_ckpt(self, ckpt_name):
		ckpt = torch.load(ckpt_name)
		self.ebm.load_state_dict(ckpt['ebm'])
		self.ebm_ema.load_state_dict(ckpt['ebm'])
		self.optimizer.load_state_dict(ckpt['optimizer'])
		step = ckpt['step']
		alpha = ckpt['alpha']
		used_sample = ckpt['used_sample']
		return step, alpha, used_sample


	# def evaluate(self, ckpt_name, sampler_step, fig_name='samples'):
	# 	with torch.no_grad():
	# 		step, alpha, _= self.load_ckpt(ckpt_name)
	# 		samples = self.langevin_sampler(x_k=x_k, sampler_step=sampler_step, step=step, alpha=alpha)
	# 		sample_path = os.path.join(os.path.dirname(ckpt_name), 'samples')
	# 		if not os.path.exists(sample_path):
	# 			os.makedirs(sample_path)
	# 		self.plot(os.path.join(sample_path, f'{fig_name}_{sampler_step}.png'), samples)
	# 		return

	@staticmethod
	def normalize_tensor(tensor):
		return tensor.add_(-tensor.min()).div_(tensor.max() - tensor.min() + 1e-5)

	def prepare_sample(self, sampler_step, im_size, step, alpha, batch_size, folder_name=None, soft=False):
		if folder_name is not None:
			path = os.path.join(self.args.run_dir, folder_name)
			if not os.path.exists(path):
				os.makedirs(path)
		samples = []
		label = None
		self.ebm.eval()
		with torch.enable_grad():
			for _ in range(30_000 // batch_size):
				x_k = self.sample_p_0(bs=batch_size, im_sz=im_size, initial=args.initial)
				if self.args.conditional:
					label = torch.randint(0, 10, size=(batch_size, ), device=self.device)
				sample = self.langevin_sampler(x_k=x_k, sampler_step=sampler_step, step=step, alpha=alpha, label=label, soft=soft, sampling=True)
				samples.append(self.normalize_tensor(sample).cpu())
		samples = torch.cat(samples, dim=0)
		return samples

	def calculate_fid(self, stats_path, sampler_step, im_size, step, alpha, soft=False, batch_size=64, dims=2048):
		samples = self.prepare_sample(sampler_step=sampler_step, im_size=im_size, step=step, alpha=alpha, batch_size=batch_size, soft=soft)
		fid_score = fid.calculate_fid_given_sample_path(samples, stats_path, batch_size, cuda=self.device, dims=dims)
		del samples
		return fid_score

	def train(self, args, dataset, save_path, load=False, ckpt_name=None):
		step = int(math.log2(args.init_size)) - 2

		if load and ckpt_name is not None:
			step, alpha, used_sample = self.load_ckpt(ckpt_name)
		else:
			used_sample = 0

		base_sigma = 1e-2
		resolution = 4 * 2 ** step
		loader = self.sample_data(
			dataset, args.batch.get(resolution, args.batch_default), resolution
		)
		data_loader = iter(loader)
		self.adjust_lr(args.lr.get(resolution, self.default_lr))
		prev_step = 0
		max_step = int(math.log2(args.max_size)) - 2
		final_progress = False

		total_used_samples = 0
		interval = 2000
		T = args.iterations
		for i in range(T):
			self.ebm.train()
			self.ebm.zero_grad()
			alpha = min(1, 1 / args.phase * (used_sample + 1))

			if (resolution == args.init_size and args.ckpt is None) or final_progress:
				alpha = 1

			if used_sample > 2 * args.phase:
				torch.save(
					{
						'used_sample': used_sample,
						'alpha': alpha,
						'step': step,
						'ebm': self.ebm.state_dict(),
						'ebm_ema': self.ebm_ema.state_dict(),
						'optimizer': self.optimizer.state_dict(),
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
				loader = self.sample_data(
					dataset, args.batch.get(resolution, args.batch_default), resolution
				)
				data_loader = iter(loader)
				self.adjust_lr(args.lr.get(resolution, self.default_lr))
			try:
				real_image, real_label = next(data_loader)

			except (OSError, StopIteration):
				data_loader = iter(loader)
				real_image, real_label = next(data_loader)

			# Sample images
			used_sample += real_image.shape[0]
			total_used_samples += real_image.shape[0]
			b_size = real_image.size(0)
			real_image, real_label = real_image.to(self.device), real_label.to(self.device)
			sigma = self.cyclic(T, i, base_sigma, M=1, min_lr=0.01) if args.cyclic else base_sigma

			x_p_d = self.sample_p_d(real_image, sigma)

			fake_label = real_label.random_(0, 10)
			if not args.conditional:
				real_label, fake_label = None, None
			if prev_step != step:
				self.plot(os.path.join(save_path, 'resolution_{}.png'.format(resolution)), x_p_d)
				prev_step += 1

			# % Langevin sampling
			if step == self.min_step or not self.progressive:
				im_size = self.resolution[self.min_step]
			else:
				if args.from_beginning:
					im_size = self.resolution[self.min_step]
				else:
					im_size = self.resolution[step - 1] if alpha < 1 or self.noise_ratio < 1 and args.soft else self.resolution[step]

			self.ebm.eval()
			with torch.enable_grad():

				x_k = self.sample_p_0(bs=b_size, im_sz=im_size, initial=args.initial)
				x_k_cp = x_k.clone().detach()
				x_q = self.langevin_sampler(x_k=x_k, sampler_step=args.langevin_step, step=step, alpha=alpha, from_beginning=args.from_beginning, soft=args.soft)
				if args.ema and i % 100 == 0 and i > args.ema_start:
					x_q_ema = self.langevin_sampler(x_k=x_k_cp, sampler_step=args.langevin_step, step=step, alpha=alpha, from_beginning=args.from_beginning, soft=args.soft,
					                            sampling=True)

			self.ebm.train()
			x = torch.cat((x_p_d, x_q), dim=0)
			label = torch.cat((real_label, fake_label), dim=0) if args.conditional else None
			energy = self.ebm(x, step=step, alpha=alpha, label=label)

			pos_energy, neg_energy = [e for e in torch.split(energy, [x_k.shape[0], x_q.shape[0]])]

			l2_regularizer = pos_energy**2 - neg_energy**2
			L = (pos_energy - neg_energy + args.l2_coefficient*l2_regularizer).mean()
			self.optimizer.zero_grad()
			(-L).backward()
			self.optimizer.step()
			#
			if abs(L.data) > 1000:
				break
			if args.ema:
				# if step == max_step and alpha == 1:
				if i > args.ema_start:
					self.ema_manager.update(decay=args.momentum)
				else:
					self.ema_manager.update(decay=0.)

			if i % 100 == 0:
				print('Itr: {:>6d}, Kimg: {:>8d}, Alpha: {:>3.2f}, Res: {:>3d}, f(x_p_d)={:8.4f}, f(x_q)={:>8.4f}'.
				      format(i, total_used_samples // 1000, alpha, resolution, pos_energy.mean(), neg_energy.mean()))

				self.plot(os.path.join(save_path, '{:>06d}_q.png'.format(i)), x_q)
				if args.ema and i > args.ema_start:
					self.plot(os.path.join(save_path, '{:>06d}_q_ema.png'.format(i)), x_q_ema)

				if i % interval == 0 and i != 0 and args.dataset in ['cifar10'] and alpha==1 and args.max_size < 128:
					print('Calculating FID score ...')

					fid_score = self.calculate_fid(stats_path=args.stats_path, im_size=im_size, sampler_step=args.langevin_step, step=step, alpha=alpha, soft=args.soft)
					print(f"Itr: {i:>6d}, FID:{fid_score:>10.3f}")
					torch.save(
						{
							'used_sample': used_sample,
							'alpha': alpha,
							'step': step,
							'ebm': self.ebm.state_dict(),
							'optimizer': self.optimizer.state_dict(),
							'ebm_ema':self.ebm_ema.state_dict(),
						},
						f'{save_path}/train_step-{i}.model',
					)
		return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
	# Set dataset
	parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/ffhq-128/celeba')
	parser.add_argument('--data_root', type=str, default="./data/", help='path of specified dataset')

	# Training stuff
	parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
	parser.add_argument('--device', type=str, default='0')


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
	parser.add_argument('--soft', action='store_true', default=True, help="if add soft connection in transition stage")
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
		dir_name = f'step{args.langevin_step}-trunc{args.truncation}-res{args.max_size}-lr{args.langevin_lr}-{args.initial}-ch{args.base_channel}-{args.activation_fn}'
		if args.ema:
			dir_name += f'-ema{args.momentum}'
		run_dir = _create_run_dir_local(save_path, dir_name)
		_copy_dir(['module', 'utils', 'train_v0.py'], run_dir)
		sys.stdout = Logger(os.path.join(run_dir, 'log.txt'))

		args.run_dir = run_dir
		transform = tr.Compose(
			[
				# tr.Resize(32),
				tr.RandomHorizontalFlip(),
				tr.ToTensor(),
				tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
			]
		)

		dataset_path = os.path.join(args.data_root, args.dataset)
		dataset = dataset_util.MultiResolutionDataset(dataset_path, transform)

		if args.sched:
			args.lr = {8: 0.0010, 16: 0.0012, 32: 0.0015, 64: 0.0015, 128: 0.0015}
			args.batch = {8: 512, 16: 256, 32: 128, 64: 64, 128: 32, 256: 32}

		else:
			args.lr = {}
			args.batch = {}
		args.batch_default = 64
		args.lr_default = 1e-3
		print(args)
		trainer = ProgressiveEBM(args)


		trainer.train(args, dataset, run_dir, load=args.load, ckpt_name=args.ckpt)

	elif args.mode == "eval" and args.ckpt is not None:
		trainer = ProgressiveEBM(args)
		trainer.evaluate(ckpt_name=args.ckpt, sampler_step=args.eval_step)
