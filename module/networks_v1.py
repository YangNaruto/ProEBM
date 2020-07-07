import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, input):
		return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
	k = torch.tensor(k, dtype=torch.float32)

	if k.ndim == 1:
		k = k[None, :] * k[:, None]

	k /= k.sum()

	return k


class Downsample(nn.Module):
	def __init__(self, kernel, factor=2):
		super().__init__()

		self.factor = factor
		kernel = make_kernel(kernel)
		self.register_buffer('kernel', kernel)

		p = kernel.shape[0] - factor

		pad0 = (p + 1) // 2
		pad1 = p // 2

		self.pad = (pad0, pad1)

	def forward(self, input):
		out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

		return out


class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor=1):
		super().__init__()

		kernel = make_kernel(kernel)

		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)

		self.register_buffer('kernel', kernel)

		self.pad = pad

	def forward(self, input):
		out = upfirdn2d(input, self.kernel, pad=self.pad)

		return out


class EqualConv2d(nn.Module):
	def __init__(
		self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
	):
		super().__init__()

		self.weight = nn.Parameter(
			torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		)
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))

		else:
			self.bias = None

	def forward(self, input):
		out = F.conv2d(
			input,
			self.weight * self.scale,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
		)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)


class EqualLinear(nn.Module):
	def __init__(
		self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
	):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

		else:
			self.bias = None

		self.activation = activation

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):
		if self.activation:
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)

		else:
			out = F.linear(
				input, self.weight * self.scale, bias=self.bias * self.lr_mul
			)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
		)


class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		out = F.leaky_relu(input, negative_slope=self.negative_slope)

		return out * math.sqrt(2)


class ConstantInput(nn.Module):
	def __init__(self, channel, size=4):
		super().__init__()

		self.input = nn.Parameter(torch.randn(1, channel, size, size))

	def forward(self, input):
		batch = input.shape[0]
		out = self.input.repeat(batch, 1, 1, 1)

		return out


class ConvLayer(nn.Sequential):
	def __init__(
		self,
		in_channel,
		out_channel,
		kernel_size,
		downsample=False,
		blur_kernel=[1, 3, 3, 1],
		bias=True,
		activate=True,
	):
		layers = []

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2
			self.padding = 0

		else:
			stride = 1
			self.padding = kernel_size // 2

		layers.append(
			EqualConv2d(
				in_channel,
				out_channel,
				kernel_size,
				padding=self.padding,
				stride=stride,
				bias=bias and not activate,
			)
		)

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))

			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(
			in_channel, out_channel, 1, downsample=True, activate=False, bias=False
		)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class Attention(nn.Module):
	""" Self attention Layer"""

	def __init__(self, in_dim, activation=None):
		super(Attention, self).__init__()
		self.chanel_in = in_dim
		self.activation = activation

		self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax = nn.Softmax(dim=-1)  #

	def forward(self, x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, width, height = x.size()
		proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
		proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm(proj_query, proj_key)  # transpose check
		attention = self.softmax(energy)  # BX (N) X (N)
		proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm(proj_value, attention.permute(0, 2, 1))
		out = out.view(m_batchsize, C, width, height)

		out = self.gamma * out + x
		return out


class Discriminator(nn.Module):
	def __init__(self, base_channel=256, fused=False, spectral=False, from_rgb_activate=False, add_attention=False, res=True, projection=True,
				 activation_fn='lrelu', bn=False):
		super().__init__()
		self.attention = nn.ModuleDict()

		if add_attention:
			self.attention  =nn.ModuleDict({'4': Attention(in_dim=base_channel * 2),
										'5': Attention(in_dim=base_channel * 2)})
		self.res = res
		if activation_fn=='lrelu':
			self.activation = nn.LeakyReLU(0.2)
		else:
			self.activation = nn.GELU()



		self.progression = nn.ModuleDict(
			{
				'0': ResBlock(16, 32, activation_fn=activation_fn),  # 512
				'1': ResBlock(32, 64, activation_fn=activation_fn),  # 256

				'2': ResBlock(base_channel//2, base_channel, activation_fn=activation_fn),  # 128
				'3': ResBlock(base_channel, base_channel, 3, 1, downsample=True, fused=fused, activation_fn=activation_fn),  # 64
				'4': ResBlock(base_channel, base_channel * 2, 3, 1, downsample=True, activation_fn=activation_fn),  # 32
				'5': ResBlock(base_channel * 2, base_channel * 2, 3, 1, downsample=True, activation_fn=activation_fn),  # 16
				'6': ConvBlock(base_channel * 2, base_channel * 4, 3, 1, downsample=True, activation_fn=activation_fn),  # 8
				'7': ConvBlock(base_channel * 4, base_channel * 4, 3, 1, downsample=True, activation_fn=activation_fn),  # 4
				'8': ConvBlock(base_channel * 4 + 1, base_channel * 4, 3, 1, 4, 0, activation_fn=activation_fn)
			}
		)
		self.final_conv = ConvBlock(base_channel * 4 + 1, base_channel * 4, 3, 1, 4, 0, activation_fn=activation_fn)

		def make_from_rgb(out_channel):
			if from_rgb_activate:
				return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

			else:
				return EqualConv2d(3, out_channel, 1)

		self.from_rgb = nn.ModuleList(
			[
				make_from_rgb(16),
				make_from_rgb(32),

				make_from_rgb(base_channel//2),
				make_from_rgb(base_channel),
				make_from_rgb(base_channel),
				make_from_rgb(base_channel*2),
				make_from_rgb(base_channel*2),
				make_from_rgb(base_channel*4),
				make_from_rgb(base_channel*4),
			]
		)

		self.norm = nn.ModuleList([nn.InstanceNorm2d(3),
								   nn.InstanceNorm2d(3),nn.InstanceNorm2d(3), nn.InstanceNorm2d(3), nn.InstanceNorm2d(3)])
		self.n_layer = len(self.progression)

		# self.linear = EqualLinear(base_channel*4, 1)
		if projection:
			self.linear = nn.Sequential(EqualLinear(base_channel*4, base_channel*16), nn.LeakyReLU(0.2), EqualLinear(base_channel*16, 1))
		else:
			self.linear = nn.Sequential(EqualLinear(base_channel*4, 1))

	def forward(self, input, step=0, alpha=-1, label=None):
		out = None
		for i in range(step, 0, -1):
			index = self.n_layer - i - 1

			if i == step:
				out = self.from_rgb[index](input)
			# if i == 0:
			# 	assert out is not None
			# 	out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
			# 	mean_std = out_std.mean()
			# 	mean_std = mean_std.expand(out.size(0), 1, 4, 4)
			# 	out = torch.cat([out, mean_std], 1)

			out = self.progression[str(index)](out, self.res)
			# if index in self.attention.keys():
			# 	out = self.attention[str(index)](out)

			if i > 0:
				if i == step and 0 <= alpha < 1:
					skip_rgb = F.avg_pool2d(input, 2)
					# skip_rgb = F.interpolate(input, scale_factor=0.5, mode='nearest')
					skip_rgb = self.from_rgb[index + 1](skip_rgb)

					out = (1 - alpha) * skip_rgb + alpha * out

		assert out is not None
		out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
		mean_std = out_std.mean()
		mean_std = mean_std.expand(out.size(0), 1, 4, 4)
		out = torch.cat([out, mean_std], 1)
		out = self.final_conv(out, res=False)
		out = out.squeeze(2).squeeze(2)
		# print(input.size(), out.size(), step)
		out = self.linear(out)

		return out