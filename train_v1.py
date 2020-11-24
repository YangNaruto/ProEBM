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
from utils.util import EMA, read_image_folder, count_parameters, read_single_image, plot_heatmap, nearest_neighbor

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

		self.resolution = [2**x for x in range(int(math.log2(args.init_size)) - 1, int(math.log2(args.max_size))+1)]
		self.create_nets(len(self.resolution)-1, args)

		self.args = args
		self.truncation = args.truncation
		self.min_step, self.max_step = int(math.log2(args.init_size)) - 2, int(math.log2(args.max_size)) - 2
		self.val_clip = args.val_clip
		self.noise_ratio = args.noise_ratio

		self.progressive = args.pro
		print("Total number of parameters: ", sum([count_parameters(self.ebm[str(step)]) for step in range(self.min_step, self.max_step+1)]))

	def switch_optimzer(self, step):
		self.optimizer = torch.optim.Adam(self.ebm[str(step)].parameters(), lr=self.args.lr_default, betas=(0.5, 0.999))

	def requires_grad(self, flag=True, step=-1):
		for p in self.ebm[str(step)].parameters():
			p.requires_grad = flag

	def create_nets(self, n, args):
		num_classes = 10 if args.dataset == 'cifar10' else 1000
		self.ebm = nn.ModuleDict({str(i): networks.EBM(base_channel=args.base_channel, spectral=args.spectral, res=args.res,
		                                  projection=args.projection, activation_fn=args.activation_fn,
		                                  from_rgb_activate=args.from_rgb_activate, bn=args.bn, split=args.split_bn,
		                                  add_attention=args.attention, num_classes=num_classes).to(self.device) for i in range(1, n+1)})

	@staticmethod
	def add_noise(x, sigma=1e-2):
		return  x + sigma * t.randn_like(x)

	def sample_data(self, dataset, batch_size, image_size=4):
		dataset.resolution = image_size
		if batch_size == -1:
			batch_size = len(dataset)
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

	def polynomial(self, t, T, base_lr, end_lr=0.0001, power=1.):
		lr = (base_lr - end_lr) * ((1 - t / T) ** power) + end_lr
		return lr

	def adjust_lr(self, lr):
		for group in self.optimizer.param_groups:
			mult = group.get('mult', 1)
			group['lr'] = lr * mult

	def langevin_sampler(self, x_k, sampler_step, step, alpha, from_beginning=False, label=None, soft=False, sampling=False):
		alphas = [1 for _ in range(step)]
		alphas.append(alpha)
		langevin_steps = sampler_step
		if step == self.min_step or not self.progressive:
			step_list = range(step, step + 1)
		else:
			if from_beginning:
				step_list = range(self.min_step, step + 1)
			else:
				step_list = range(step - 1, step + 1)

		step_list = list(step_list)
		K = 0
		for index, s in enumerate(step_list):
			if index != 0:
				x_k = F.interpolate(x_k.clone().detach(), scale_factor=2, mode='nearest')
				x_k = torch.clamp(x_k, -self.val_clip, self.val_clip)

			x_k.requires_grad_(True)

			for k in range(langevin_steps):
				loss = self.ebm[str(s)](x_k, step=s, alpha=alphas[s], label=label).sum()
				f_prime = t.autograd.grad(loss, [x_k], retain_graph=True)[0]

				sigma = self.polynomial(K, langevin_steps*len(step_list), base_lr=1.5e-2, end_lr=0e-4, power=1) if self.args.cyclic else 5e-3
				lr = self.polynomial(k, langevin_steps, base_lr=self.args.langevin_lr, end_lr=self.args.langevin_lr, power=1) if self.args.cyclic else self.args.langevin_lr

				x_k.data += lr * f_prime + sigma * t.randn_like(x_k)
				K += 1
			x_k.data.clamp_(-self.val_clip, self.val_clip)

		return x_k.clone()


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
		tv.utils.save_image(x[:row ** 2], p, padding=0, normalize=True, range=(-1., 1.), nrow=row)
		return

	def load_ckpt(self, ckpt_name):
		ckpt = torch.load(ckpt_name)
		step = ckpt['step']
		alpha = ckpt['alpha']
		for i in range(step):
			self.ebm[str(i+1)].load_state_dict(ckpt['ebm-%d'%i])

		used_sample = ckpt['used_sample']
		return step, alpha, used_sample

	def test_loader(self, dataset, dataset_name='cifar10', batch_size=64):
		transform = tr.Compose(
			[tr.ToTensor(),
			 tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		if dataset_name == 'cifar10':
			dataset = tv.datasets.CIFAR10('./data', train=True, transform=transform, target_transform=None, download=False)
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
		else:
			data_loader = self.sample_data(dataset, batch_size, 64)

		return data_loader

	def find_knn(self, dataset, batch_size=5, k=10, step=3):
		im_size = self.resolution[step]
		if not os.path.exists('knn'):
			os.mkdir('knn')
		else:
			shutil.rmtree('knn')
			os.mkdir('knn')
		data_loader = self.test_loader(dataset, dataset_name=args.dataset, batch_size=10000)
		dataiter = iter(data_loader)
		images, _ = next(dataiter)
		images = torch.flatten(images, start_dim=1).unsqueeze(0)
		images = images.to(self.device)
		for i in range(500):
			x0 = self.sample_p_0(batch_size, im_size, n_ch=3, initial=self.args.initial).clone().detach()
			samples = self.langevin_sampler(x_k=x0, sampler_step=self.args.langevin_step, step=step, alpha=1, label=None, soft=True,
												 sampling=False)
			samples = torch.flatten(samples, start_dim=1).unsqueeze(1)

			nearest_neighbor(samples, images, i=i, k=10, logdir='knn', im_size=im_size)


	def inpainting(self, dataset, args, batch_size=64, step=3):
		if not os.path.exists('inpainting'):
			os.mkdir('inpainting')
		else:
			shutil.rmtree('inpainting')
			os.mkdir('inpainting')
		row = int(sqrt(batch_size))

		data_loader = self.test_loader(dataset, dataset_name=args.dataset, batch_size=batch_size)
		dataiter = iter(data_loader)
		im_size = self.resolution[step]
		mask = torch.ones((batch_size, 3, im_size, im_size), device=self.device)
		index = torch.randint(0, 64, (batch_size, 8, 64), device=self.device)
		mask[:, 0].scatter_(1, index, 0)
		mask[:, 1].scatter_(1, index, 0)
		mask[:, 2].scatter_(1, index, 0)

		for i in range(1000):
			images, labels = next(dataiter)
			images = images.to(self.device)
			noise = torch.zeros_like(mask)* (1 - mask)

			x = images.clone().detach() * mask + noise
			x_masked = x.clone().detach()
			for _ in range(int(self.args.langevin_step/2)):
				x = self.langevin_sampler(x_k=x, sampler_step=1, step=step, alpha=1, label=None, soft=True, sampling=False)
				x = x * (1 - mask) + images * mask
			group = torch.cat((x_masked, x, images), dim=0)
			tv.utils.save_image(group, 'inpainting/sample_{:02d}.png'.format(i), padding=0, normalize=True,
									range=(-1., 1.),
									nrow=batch_size)

	def interpolation(self, step, batch_size=4):
		if not os.path.exists('interpolation'):
			os.mkdir('interpolation')
		else:
			shutil.rmtree('interpolation')
			os.mkdir('interpolation')
		if step == self.min_step or not self.progressive:
			im_size = self.resolution[self.min_step]
		else:
			if self.args.from_beginning:
				im_size = self.resolution[self.min_step]
			else:
				im_size = self.resolution[step - 1]

		beta = list(range(0, 11, 1))
		for i in range(500):
			x0 = self.sample_p_0(batch_size, im_size, n_ch=3, initial=self.args.initial).clone().detach()
			x1 = self.sample_p_0(batch_size, im_size, n_ch=3, initial=self.args.initial).clone().detach()

			sequence = torch.zeros((batch_size, 11, 3, self.resolution[step], self.resolution[step])).to(self.device)
			for j in range(0, 11):
				x = np.sqrt(1 - (beta[j]/10)**2) * x0 + beta[j]/10 * x1
				images = self.langevin_sampler(x_k=x, sampler_step=self.args.langevin_step, step=step, alpha=1, label=None, soft=True,
													 sampling=False, from_beginning=self.args.from_beginning)
				sequence[:, j] = images
			tv.utils.save_image(sequence.view(batch_size*11, 3, self.resolution[step], self.resolution[step]), 'interpolation/seq_{:02d}.png'.format(i), padding=0, normalize=True,
								range=(-1., 1.),
								nrow=11)


	def from_noise(self,  step, batch_size=4):
		path = 'samples/church'
		if not os.path.exists(path):
			os.makedirs(path)
		else:
			shutil.rmtree(path)
			os.makedirs(path)
		# im_size = self.resolution[step]
		if step == self.min_step or not self.progressive:
			im_size = self.resolution[self.min_step]
		else:
			if self.args.from_beginning:
				im_size = self.resolution[self.min_step]
			else:
				im_size = self.resolution[step - 1]

		row = int(sqrt(batch_size))
		for i in range(5000):
			x0 = self.sample_p_0(batch_size, im_size, n_ch=3, initial=self.args.initial).clone().detach()
			images = self.langevin_sampler(x_k=x0, sampler_step=self.args.langevin_step, step=step, alpha=1, label=None, soft=True,
												 sampling=False, from_beginning=self.args.from_beginning)


			tv.utils.save_image(images, os.path.join(path, 'sample_{:03d}.png'.format(i)), padding=0, normalize=True,
							   range=(-1., 1.),
								nrow=row)
			# interval = 5
			# seq_len = images[:, ::interval].shape[1]
			# tv.utils.save_image(images[:, ::interval].view(batch_size*seq_len, 3, im_size, im_size), 'from_noise/seq_{:02d}.png'.format(i), padding=0, normalize=True,
			# 					range=(-1., 1.),
			# 					nrow=seq_len)
			print(f'Saving {i}-th image...')

	def eval_fid(self, step, batch_size=64):
		if step == self.min_step or not self.progressive:
			im_size = self.resolution[self.min_step]
		else:
			if self.args.from_beginning:
				im_size = self.resolution[self.min_step]
			else:
				im_size = self.resolution[step - 1]

		samples = []
		for _ in range(3_000 // batch_size):

			x0 = self.sample_p_0(batch_size, im_size, n_ch=3, initial=self.args.initial).clone().detach()
			sample = self.langevin_sampler(x_k=x0, sampler_step=self.args.langevin_step, step=step, alpha=1, label=None, soft=True,
												 sampling=False, from_beginning=self.args.from_beginning)
			samples.append(self.normalize_tensor(sample).cpu())
		samples = torch.cat(samples, dim=0)

		fid_score = fid.calculate_fid_given_sample_path(samples, "./stats/celeba-hq_128.npz", batch_size//4, cuda=self.device, dims=2048)
		del samples
		print(fid_score)
		return

	def anti_corrupt(self, dataset, args, batch_size=4, step=3):
		if not os.path.exists('anti_corrupt'):
			os.mkdir('anti_corrupt')
		else:
			shutil.rmtree('anti_corrupt')
			os.mkdir('anti_corrupt')
		row = int(sqrt(batch_size))
		interval = 1
		data_loader = self.test_loader(dataset, dataset_name=args.dataset, batch_size=batch_size)
		dataiter = iter(data_loader)
		for i in range(200):

			images, labels = next(dataiter)
			images = images.to(self.device)

			noise = torch.rand_like(images)
			low_mask = noise < 1.0
			# high_mask = (noise > 0.05) & (noise < 0.1)

			images_corrupt = images.clone()
			# images_corrupt[low_mask] = -0.8
			# images_corrupt[high_mask] = 0.8
			images_corrupt[low_mask] = images_corrupt[low_mask]  + 0.2*torch.randn_like(images_corrupt[low_mask] )
			# images_corrupt[high_mask] = images_corrupt[high_mask] + 0.3*torch.randn_like(images_corrupt[high_mask])
			# images_corrupt = torch.clamp(images_corrupt, -1, 1)
			x0 = images_corrupt.clone().detach()
			clean_images, sample_seq  = self.langevin_sampler(x_k=x0, sampler_step=10, step=step, alpha=1, label=None, soft=True,
														 sampling=False)
			group = torch.cat((images_corrupt, clean_images, images), dim=0)
			tv.utils.save_image(group, 'anti_corrupt/group_{:02d}.png'.format(i), padding=0, normalize=True,
							   range=(-1., 1.),
								nrow=3)
			plot_heatmap(sample_seq.cpu().numpy(), fig_name=os.path.join('anti_corrupt', 'group_{:02d}_heatmap.png'.format(i)))

			seq_path = os.path.join('anti_corrupt',  'group_{:02d}_seq.png'.format(i))
			seq_len = sample_seq[:, ::interval].shape[1]
			tv.utils.save_image(sample_seq[:, ::interval].view(seq_len, 3, 32, 32), seq_path, padding=0, normalize=True, range=(-1., 1.),
			                    nrow=seq_len)
		# tv.utils.save_image(images[:row ** 2], 'anti_corrupt/raw_anticorrupt.png', padding=0, normalize=True, range=(-1., 1.),
		# 					nrow=row)
		# tv.utils.save_image(images_corrupt[:row ** 2], 'anti_corrupt/corrupt.png', padding=0, normalize=True, range=(-1., 1.),
		# 					nrow=row)
		# tv.utils.save_image(clean_images[:row ** 2], 'anti_corrupt/clean_{:02d}.png'.format(i), padding=0,
		# 					normalize=True, range=(-1., 1.),
		# 					nrow=row)
		# for i, img in enumerate(clean_images):
		# 	tv.utils.save_image(clean_images[i][:row ** 2], 'anti_corrupt/clean_{:02d}.png'.format(i), padding=0, normalize=True, range=(-1., 1.),
		# 					nrow=row)

	@staticmethod
	def normalize_tensor(tensor):
		return tensor.add_(-tensor.min()).div_(tensor.max() - tensor.min() + 1e-5)

	def prepare_sample(self, sampler_step, im_size, step, alpha, batch_size, folder_name=None, soft=False, from_beginning=False):
		if folder_name is not None:
			path = os.path.join(self.args.run_dir, folder_name)
			if not os.path.exists(path):
				os.makedirs(path)
		samples = []
		label = None
		self.ebm[str(step)].eval()
		with torch.enable_grad():
			for _ in range(30_000 // batch_size):
				x_k = self.sample_p_0(bs=batch_size, im_sz=im_size, initial=self.args.initial)
				if self.args.conditional:
					label = torch.randint(0, 10, size=(batch_size, ), device=self.device)
				sample = self.langevin_sampler(x_k=x_k, sampler_step=sampler_step, step=step, alpha=alpha, label=label, soft=soft, sampling=True, from_beginning=from_beginning)
				samples.append(self.normalize_tensor(sample).cpu())
		samples = torch.cat(samples, dim=0)
		return samples

	def calculate_fid(self, stats_path, sampler_step, im_size, step, alpha, from_beginning=False, soft=False, batch_size=64, dims=2048):
		samples = self.prepare_sample(sampler_step=sampler_step, im_size=im_size, step=step, alpha=alpha, batch_size=batch_size, soft=soft, from_beginning=from_beginning)
		fid_score = fid.calculate_fid_given_sample_path(samples, stats_path, batch_size//4, cuda=self.device, dims=dims)
		del samples
		return fid_score

	def train(self, args, dataset, save_path, load=False, ckpt_name=None):
		step = int(math.log2(args.init_size)) - 2

		if load and ckpt_name is not None:
			step, alpha, used_sample = self.load_ckpt(ckpt_name)
		else:
			used_sample = 0

		base_sigma = args.base_sigma
		resolution = 4 * 2 ** step
		loader = self.sample_data(
			dataset, args.batch.get(resolution, args.batch_default), resolution
		)
		data_loader = iter(loader)
		self.switch_optimzer(step)
		self.adjust_lr(args.lr.get(resolution, self.default_lr))
		prev_step = 0
		max_step = int(math.log2(args.max_size)) - 2
		final_progress = False

		total_used_samples = 0
		interval = 2000
		T = args.iterations
		for i in range(T):
			self.ebm[str(step)].train()
			self.ebm[str(step)].zero_grad()
			alpha = min(1, 1 / args.phases.get(resolution, args.phase) * (used_sample + 1))

			if (resolution == args.init_size and args.ckpt is None) or final_progress:
				alpha = 1

			if used_sample > 2 * args.phases.get(resolution, args.phase):

				used_sample = 0
				step += 1
				if step <= max_step:
					self.switch_optimzer(step)
					self.ebm[str(step)].load_state_dict(self.ebm[str(step-1)].state_dict())

				if step > max_step:
					step = max_step
					final_progress = True
				else:
					alpha = 0

				resolution = 4 * 2 ** step

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
			sigma = base_sigma

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
					im_size = self.resolution[step - 1]

			self.ebm[str(step)].eval()
			with torch.enable_grad():
				x_k = self.sample_p_0(bs=b_size, im_sz=im_size, initial=args.initial)
				x_q = self.langevin_sampler(x_k=x_k, sampler_step=args.langevin_step, step=step, alpha=alpha, from_beginning=args.from_beginning, soft=args.soft)

			self.ebm[str(step)].train()
			x = torch.cat((x_p_d, x_q), dim=0)
			label = torch.cat((real_label, fake_label), dim=0) if args.conditional else None
			energy =self.ebm[str(step)](x, step=step, alpha=alpha, label=label)

			pos_energy, neg_energy = [e for e in torch.split(energy, [x_k.shape[0], x_q.shape[0]])]

			l2_regularizer = pos_energy**2 - neg_energy**2
			L = (pos_energy - neg_energy + args.l2_coefficient*l2_regularizer).mean()
			self.optimizer.zero_grad()
			(-L).backward()
			self.optimizer.step()

			if i % 400 == 0:
				print('Itr: {:>6d}, Kimg: {:>8d}, Alpha: {:>3.2f}, Res: {:>3d}, f(x_p_d)={:8.4f}, f(x_q)={:>8.4f}'.
				      format(i, total_used_samples // 1000, alpha, resolution, pos_energy.mean(), neg_energy.mean()))
				if abs(L.data) > 1000:
					break
				self.plot(os.path.join(save_path, '{:>06d}_q.png'.format(i)), x_q)
				if i % interval == 0 and i != 0:
					if args.dataset in ['cifar10', 'imagenet', 'celeba', 'celeba-c', 'imagenet32x32'] and alpha==1 and args.max_size < 128:
					# if args.dataset in ['imagenet32x32']:
						print('Calculating FID score ...')

						fid_score = self.calculate_fid(stats_path=args.stats_path, im_size=im_size, sampler_step=args.langevin_step, from_beginning=args.from_beginning,
						                               step=step, alpha=alpha, soft=args.soft)
						print(f"Itr: {i:>6d}, FID:{fid_score:>10.3f}")
					model_dict = {'ebm-%d' % i: self.ebm[str(i + 1)].state_dict() for i in range(step)}
					model_dict.update({
						'used_sample': used_sample,
						'alpha': alpha,
						'step': step,
						# 'optimizer': self.optimizer.state_dict(),
					})
					torch.save(model_dict,
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

	if args.sched:
		args.lr = {8: 0.0005, 16: 0.0005, 32: 0.0005, 64: 0.0007, 128: 0.0010, 256: 0.0015, 512: 0.0015}
		args.batch = {8: 256, 16: 128, 32: 64, 64: 64, 128: 32, 256: 16, 512: 16}
		# args.phases = {8:args.phase, 16:args.phase*1.5, 32: args.phase*2, 64: args.phase*2, 128: args.phase*2, 256: args.phase*2, }
		args.phases = {8:args.phase, 16:args.phase, 32: args.phase, 64: args.phase, 128: args.phase, 256: args.phase, 512: args.phase}
		# if args.dataset in ['imagenet32x32', 'imagenet']:
			# for key, value in args.batch.items():
			# 	args.batch.update({key: int(value * 2)})
		# args.batch = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32, 256: 32}
		args.lr = {key: value + 5e-4 for key, value in args.lr.items()}
	else:
		args.lr = {}
		args.batch = {}
	args.batch_default = 32
	args.lr_default = 5e-4

	if args.mode == "train":
		dir_name = f'step{args.langevin_step}-trunc{args.truncation}-res{args.max_size}-lr{args.langevin_lr}-{args.initial}-ch{args.base_channel}-{args.activation_fn}'
		if args.ema:
			dir_name += f'-ema{args.momentum}'
		run_dir = _create_run_dir_local(save_path, dir_name)
		_copy_dir(['module', 'utils', 'train_v1.py'], run_dir)
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


		print(args)
		trainer = ProgressiveEBM(args)


		trainer.train(args, dataset, run_dir, load=args.load, ckpt_name=args.ckpt)


	elif args.mode == "eval" and args.ckpt is not None:
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

		trainer = ProgressiveEBM(args)

		step, alpha, num = trainer.load_ckpt(ckpt_name=args.ckpt)

		# trainer.find_knn(dataset, batch_size=2, step=step, k=10)
		# trainer.interpolation(batch_size=1, step=step)
		trainer.from_noise(batch_size=1, step=step)
		# trainer.eval_fid(step, batch_size=100)
		# trainer.inpainting(dataset, args, batch_size=1, step=step)

		# trainer.anti_corrupt(dataset, args, batch_size=1, step=step)

		# trainer.denoise(dataset_name='cifar10', batch_size=64)

		# trainer.super_resolution(dataset_name='cifar10', batch_size=64)
