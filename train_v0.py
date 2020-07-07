import torch as t
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

sys.path.append("./utils/score")
from scipy.stats import truncnorm
from module import networks
from utils import dataset_util
from utils.logger import Logger
from utils.submit import _get_next_run_id_local, _create_run_dir_local, _copy_dir, _save_args
from utils.score import fid

seed = 1
sigma = 1e-2  # decrease until training is unstable
t.manual_seed(seed)
if t.cuda.is_available():
	t.cuda.manual_seed_all(seed)
noise = lambda x: x + sigma * t.randn_like(x)
sqrt = lambda x: int(t.sqrt(t.Tensor([x])))


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
class ProgressiveEBM(object):
	def __init__(self, args):
		super().__init__()
		self.default_lr = args.lr_default
		self.device = t.device(args.device if torch.cuda.is_available() else 'cpu')
		self.ebm = networks.Discriminator(base_channel=args.base_channel, spectral=args.spectral, res=args.res,
		                                  projection=args.projection, activation_fn=args.activation_fn,
		                                  from_rgb_activate=args.from_rgb_activate, bn=args.bn).to(self.device)
		self.args = args
		self.truncation = args.truncation
		self.min_step, self.max_step = int(math.log2(args.init_size)) - 2, int(math.log2(args.max_size)) - 2
		self.val_clip = args.val_clip
		self.noise_ratio = args.noise_ratio
		self.resolution = [2 ** x for x in range(2, 10)]
		self.progressive = args.pro
		self.optimizer = torch.optim.Adam(self.ebm.parameters(), lr=self.default_lr, betas=(0.0, 0.99))

	def requires_grad(self, flag=True):
		for p in self.ebm.parameters():
			p.requires_grad = flag

	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

	def metropolis_hastings(self, x, x_prev, step, alpha):
		energy = self.ebm(x, step=step, alpha=alpha)
		energy_prev = self.ebm(x_prev, step=step, alpha=alpha)
		acceptance = (torch.exp(-energy) / torch.exp(-energy_prev) - energy.uniform_(0., 1.)) >= 0
		x = torch.where(acceptance.view(-1, 1, 1, 1), x, x_prev)
		return x.data

	def sample_data(self, dataset, batch_size, image_size=4):
		dataset.resolution = image_size
		loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True, pin_memory=True)
		return loader

	@staticmethod
	def truncated_normal(size, threshold=1.):
		values = truncnorm.rvs(-threshold, threshold, size=size)
		return values

	def cyclic(self, T, i, M=4, min_lr=0.):
		rcounter = T + i
		cos_inner = np.pi * (rcounter % (T // M))
		cos_inner /= T // M
		cos_out = np.cos(cos_inner) + 1
		lr = np.clip(0.5 * cos_out * self.args.langevin_lr, min_lr, 100)
		return lr

	def adjust_lr(self, lr):
		for group in self.optimizer.param_groups:
			mult = group.get('mult', 1)
			group['lr'] = lr * mult

	def langevin_sampler(self, x_k, sampler_step, step, alpha, from_beginning=False, label=None):
		alphas = [1 for _ in range(step)]
		alphas.append(alpha)

		if step == self.min_step or not self.progressive:
			step_list = range(step, step + 1)
		else:
			if from_beginning:
				step_list = range(self.min_step, step + 1)
			else:
				step_list = range(step - 1, step + 1) if alpha < 1 or self.noise_ratio < 1 else range(step, step+1)

		# initial = self.args.initial if self.args.noise_ratio == 1.0 else 'gaussian'
		step_list = list(step_list)
		x_k.requires_grad_(True)
		for index, s in enumerate(step_list):
			if index != 0:
				# x_k = self.ebm.norm[s](x_k)
				x_k = F.interpolate(x_k, scale_factor=2, mode='nearest')
				x_k = (1 - self.noise_ratio * alphas[s]) * x_k + \
				      self.noise_ratio * alphas[s] * self.sample_p_0(x_k.shape[0], x_k.shape[-1], n_ch=3, initial=self.args.initial)

				# x_k = torch.clamp(x_k, -self.val_clip, self.val_clip)

			# langevin_steps = int(sampler_step / len(step_list))
			langevin_steps = sampler_step

			for k in range(langevin_steps):
				f_prime = t.autograd.grad(self.ebm(x_k, step=s, alpha=alphas[s], label=label).sum(), [x_k], retain_graph=True)[0]
				lr = self.cyclic(langevin_steps, k) if self.args.cyclic else self.args.langevin_lr
				x_k.data += lr * f_prime + 1e-2 * t.randn_like(x_k)
				# x_k.data.clamp_(-self.val_clip, self.val_clip)

		return x_k.detach()

	def sample_p_0(self, bs, im_sz, n_ch=3, initial='gaussian'):
		if initial == 'gaussian':
			x_k = torch.from_numpy(self.truncated_normal(size=(bs, n_ch, im_sz, im_sz), threshold=self.truncation).astype(np.float32)).to(self.device)
		else:
			x_k = (-2. * torch.rand(bs, n_ch, im_sz, im_sz) + 1.).to(self.device)

		return x_k

	@staticmethod
	def sample_p_d(img):
		return noise(img).detach()

	@staticmethod
	def plot(p, x):
		m = x.shape[0]
		row = int(sqrt(m))
		tv.utils.save_image(x[:row ** 2], p, padding=1, normalize=True, range=(-1., 1.), nrow=row)
		return

	def load_ckpt(self, ckpt_name):
		ckpt = torch.load(ckpt_name)
		self.ebm.load_state_dict(ckpt['ebm'])
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

	def prepare_sample(self, sampler_step, im_size, step, alpha, batch_size, folder_name=None):
		if folder_name is not None:
			path = os.path.join(self.args.run_dir, folder_name)
			if not os.path.exists(path):
				os.makedirs(path)
		samples = []
		self.ebm.eval()
		for _ in range(30_000 // batch_size):
			x_k = self.sample_p_0(bs=batch_size, im_sz=im_size, initial=args.initial)
			sample = self.langevin_sampler(x_k=x_k, sampler_step=sampler_step, step=step, alpha=alpha)
			samples.append(self.normalize_tensor(sample).cpu())
		samples = torch.cat(samples, dim=0)
		return samples

	def calculate_fid(self, stats_path, sampler_step, im_size, step, alpha, batch_size=64, dims=2048):
		samples = self.prepare_sample(sampler_step=sampler_step, im_size=im_size, step=step, alpha=alpha, batch_size=batch_size)
		fid_score = fid.calculate_fid_given_sample_path(samples, stats_path, batch_size, cuda=self.device, dims=dims)
		del samples
		return fid_score

	def train(self, args, dataset, save_path, load=False, ckpt_name=None):
		step = int(math.log2(args.init_size)) - 2

		if load and ckpt_name is not None:
			step, alpha, used_sample = self.load_ckpt(ckpt_name)
		else:
			used_sample = 0

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
		for i in range(1_000_000):
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
					interval = 5000
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
			x_p_d = self.sample_p_d(real_image)

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
					im_size = self.resolution[step - 1] if alpha < 1 or self.noise_ratio < 1 else self.resolution[step]

			x_k = self.sample_p_0(bs=b_size, im_sz=im_size, initial=args.initial)
			x_q = self.langevin_sampler(x_k=x_k, sampler_step=args.langevin_step, step=step, alpha=alpha, from_beginning=args.from_beginning)

			x = torch.cat((x_p_d, x_q), dim=0)
			label = torch.cat((real_label, fake_label), dim=0) if args.conditional else None
			energy = self.ebm(x, step=step, alpha=alpha, label=label)
			# print(energy.shape)
			pos_energy, neg_energy = [e.mean() for e in torch.split(energy, [x_k.shape[0], x_q.shape[0]])]
			# pos_energy = self.ebm(x_p_d, step=step, alpha=alpha, label=real_label).mean()
			# neg_energy = self.ebm(x_q, step=step, alpha=alpha, label=fake_label).mean()
			L = pos_energy - neg_energy
			self.optimizer.zero_grad()
			(-L).backward()
			self.optimizer.step()
			if i % 100 == 0:
				print('Itr: {:>6d} Kimg: {:>8d}, Alpha: {:>3.2f}, Res: {:>3d}, f(x_p_d)={:8.4f} f(x_q)={:>8.4f}'.
				      format(i, total_used_samples // 1000, alpha, resolution, pos_energy, neg_energy))
				self.plot(os.path.join(save_path, '{:>06d}_q.png'.format(i)), x_q)
				if i % interval == 0 and i != 0:
					print('Calculate FID...')
					self.ebm.eval()
					fid_score = self.calculate_fid(stats_path=args.stats_path, im_size=im_size, sampler_step=args.langevin_step, step=step, alpha=alpha)
					print(f"Itr: {i:>6d}, FID:{fid_score:>10.3f}")
					torch.save(
						{
							'used_sample': used_sample,
							'alpha': alpha,
							'step': step,
							'ebm': self.ebm.state_dict(),
							'optimizer': self.optimizer.state_dict(),
						},
						f'{save_path}/train_step-{i}.model',
					)
					self.ebm.train()
		return


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
	parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/ffhq-128/celeba')
	parser.add_argument('--data_root', type=str, default="./data/", help='path of specified dataset')
	parser.add_argument('--phase', type=int, default=600_000, help='number of samples used for each training phases')
	parser.add_argument('--device', type=str, default='0')
	parser.add_argument('--spectral', action='store_true', default=False, help="add spectral normalization")
	parser.add_argument('--bn', action='store_true', default=False, help="add split batch normalization")
	parser.add_argument('--conditional', action='store_true', default=False, help="add spectral normalization")
	parser.add_argument('--mode', type=str, default='train', help="train/eval")
	parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
	parser.add_argument('--sched', action='store_true', help='use lr scheduling')
	parser.add_argument('--init_size', default=8, type=int, help='initial image size')
	parser.add_argument('--max_size', default=32, type=int, help='max image size')
	parser.add_argument('--val_clip', default=1.5, type=float, help='x clip')
	parser.add_argument('--noise_ratio', default=0.5, type=float, help='noise amount')
	parser.add_argument('--res', action='store_true', default=False)
	parser.add_argument('--projection', action='store_true', default=False)
	parser.add_argument('--from_beginning', action='store_true', default=False)

	parser.add_argument('--initial', default='uniform', type=str, help='save path')
	parser.add_argument('--langevin_step', default=50, type=int, help='langevin update steps')
	parser.add_argument('--langevin_lr', default=2., type=float, help='langevin learning rate')
	parser.add_argument('--pro', action='store_true', default=False, help='if progressivly fading')
	parser.add_argument('--activation_fn', type=str, default='lrelu', help='activation function')

	# save stuff
	parser.add_argument('--name_suffix', default='', type=str, help='save path')
	parser.add_argument('--load', action='store_true', default=False, help='if progressivly fading')
	parser.add_argument('--eval_step', default=50, type=int, help='evaluate step')
	parser.add_argument('--run_dir', default='', type=str, help='running directory')
	parser.add_argument('--stats_path', default='./stats/fid_stats_cifar10_train.npz', type=str, help='running directory')
	parser.add_argument('--truncation', default=1.0, type=float, help="truncation threshold")
	parser.add_argument('--cyclic', action='store_true', default=False, help='if apply cyclic langevin learning rate')
	parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
	parser.add_argument('--base_channel', default=128, type=int, help='the base number of filters')
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
		dir_name = f'step{args.langevin_step}-trunc{args.truncation}-res{args.max_size}-lr{args.langevin_lr}-{args.initial}-ch{args.base_channel}'
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
			args.batch = {8: 256, 16: 128, 32: 64, 64: 64, 128: 32, 256: 32}
			# args.phases = {8:600_000, 16:2_000_000, 32: 4_000_000, 64: 10_000_000}
			if args.dataset == 'metfaces':
				args.batch = {8: 64, 16: 64, 32: 32, 64: 32, 128: 16, 256: 16}
		# for key, value in args.batch.items():
		# 	args.batch.update({key: int(value * 64 / args.base_channel)})
		# args.batch = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32, 256: 32}
		# args.lr = {key : value/4 for key, value in lr.items()}
		else:
			args.lr = {}
			args.batch = {}
		args.batch_default = 50
		args.lr_default = 1e-4
		print(args)
		trainer = ProgressiveEBM(args)
		trainer.train(args, dataset, run_dir, load=args.load, ckpt_name=args.ckpt)

	elif args.mode == "eval" and args.ckpt is not None:
		trainer = ProgressiveEBM(args)
		trainer.evaluate(ckpt_name=args.ckpt, sampler_step=args.eval_step)
