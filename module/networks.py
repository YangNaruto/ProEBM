import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm
from math import sqrt
import math
from .split_bn import SplitBatchNorm2d
from .swish import SwishModule


def init_linear(linear):
	init.xavier_normal(linear.weight)
	linear.bias.data.zero_()


def init_conv(conv, glu=True):
	init.kaiming_normal(conv.weight)
	if conv.bias is not None:
		conv.bias.data.zero_()

def get_activation(activation_fn):
	if activation_fn == 'lrelu':
		return nn.LeakyReLU(0.2)
	elif activation_fn == 'gelu':
		return nn.GELU()
	elif activation_fn == 'swish':
		return SwishModule()
	elif activation_fn == 'elu':
		return nn.ELU(alpha=1.0)
	elif activation_fn == 'celu':
		return nn.CELU(alpha=1.1)
	elif activation_fn == 'relu':
		return nn.ReLU()
	elif activation_fn == 'tanh':
		return nn.Tanh()


class EqualLR:
	def __init__(self, name):
		self.name = name

	def compute_weight(self, module):
		weight = getattr(module, self.name + '_orig')
		fan_in = weight.data.size(1) * weight.data[0][0].numel()

		return weight * sqrt(2 / fan_in)

	@staticmethod
	def apply(module, name):
		fn = EqualLR(name)

		weight = getattr(module, name)
		del module._parameters[name]
		module.register_parameter(name + '_orig', nn.Parameter(weight.data))
		module.register_forward_pre_hook(fn)

		return fn

	def __call__(self, module, input):
		weight = self.compute_weight(module)
		setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
	EqualLR.apply(module, name)

	return module


class FusedUpsample(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, padding=0):
		super().__init__()

		weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
		bias = torch.zeros(out_channel)

		fan_in = in_channel * kernel_size * kernel_size
		self.multiplier = sqrt(2 / fan_in)

		self.weight = nn.Parameter(weight)
		self.bias = nn.Parameter(bias)

		self.pad = padding

	def forward(self, input):
		weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
		weight = (
						 weight[:, :, 1:, 1:]
						 + weight[:, :, :-1, 1:]
						 + weight[:, :, 1:, :-1]
						 + weight[:, :, :-1, :-1]
				 ) / 4

		out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

		return out


class FusedDownsample(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, padding=0):
		super().__init__()

		weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		bias = torch.zeros(out_channel)

		fan_in = in_channel * kernel_size * kernel_size
		self.multiplier = sqrt(2 / fan_in)

		self.weight = nn.Parameter(weight, requires_grad=True)
		self.bias = nn.Parameter(bias, requires_grad=True)

		self.pad = padding

	def forward(self, x):
		weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
		weight = (
						 weight[:, :, 1:, 1:]
						 + weight[:, :, :-1, 1:]
						 + weight[:, :, 1:, :-1]
						 + weight[:, :, :-1, :-1]
				 ) / 4

		out = F.conv2d(x, weight, self.bias, stride=2, padding=self.pad)

		return out


def pixel_norm(x):
	return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
	@staticmethod
	def forward(ctx, grad_output, kernel, kernel_flip):
		ctx.save_for_backward(kernel, kernel_flip)

		grad_input = F.conv2d(
			grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
		)

		return grad_input

	@staticmethod
	def backward(ctx, gradgrad_output):
		kernel, kernel_flip = ctx.saved_tensors

		grad_input = F.conv2d(
			gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
		)

		return grad_input, None, None


class BlurFunction(Function):
	@staticmethod
	def forward(ctx, input, kernel, kernel_flip):
		ctx.save_for_backward(kernel, kernel_flip)

		output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

		return output

	@staticmethod
	def backward(ctx, grad_output):
		kernel, kernel_flip = ctx.saved_tensors

		grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

		return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
	def __init__(self, channel):
		super().__init__()

		weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
		weight = weight.view(1, 1, 3, 3)
		weight = weight / weight.sum()
		weight_flip = torch.flip(weight, [2, 3])

		self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
		self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

	def forward(self, input):
		return blur(input, self.weight, self.weight_flip)
	# return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__()

		conv = nn.Conv2d(*args, **kwargs)
		conv.weight.data.normal_()
		if conv.bias is not None:
			conv.bias.data.zero_()
		self.conv = equal_lr(conv)

	def forward(self, input):
		return self.conv(input)


class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()

		linear = nn.Linear(in_dim, out_dim)
		linear.weight.data.normal_()
		linear.bias.data.zero_()

		self.linear = equal_lr(linear)

	def forward(self, input):
		return self.linear(input)


class ConvBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			padding,
			kernel_size2=None,
			padding2=None,
			downsample=False,
			fused=False,
			activation_fn='lrelu'
	):
		super().__init__()

		self.activation = get_activation(activation_fn=activation_fn)

		pad1 = padding
		pad2 = padding
		if padding2 is not None:
			pad2 = padding2

		kernel1 = kernel_size
		kernel2 = kernel_size
		if kernel_size2 is not None:
			kernel2 = kernel_size2

		self.conv1 = nn.Sequential(
			EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
		)

		if downsample:
			if fused:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
				)

			else:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
					nn.AvgPool2d(2),
				)
			self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1), nn.AvgPool2d(2))
		else:
			self.conv2 = nn.Sequential(
				EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
			)
			self.skip = EqualConv2d(in_channel, out_channel, 1)

		# self.skip = EqualConv2d(in_channel, out_channel, 1, stride=2, padding=0, bias=False)

	def forward(self, input, res=True):
		out = self.conv1(input)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.activation(out)
		if res:
			skip = self.skip(input)
			out = (skip + out ) / math.sqrt(2)


		return out


class BNConvBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			padding,
			kernel_size2=None,
			padding2=None,
			downsample=False,
			fused=False,
			activation_fn='lrelu',
			split=False
	):
		super().__init__()

		pad1 = padding
		pad2 = padding
		if padding2 is not None:
			pad2 = padding2

		kernel1 = kernel_size
		kernel2 = kernel_size
		if kernel_size2 is not None:
			kernel2 = kernel_size2

		self.conv1 = nn.Sequential(
			EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
		)

		if downsample:
			if fused:
				self.conv2 = nn.Sequential(
					Blur(out_channel),
					FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),

				)

			else:
				self.conv2 = nn.Sequential(
					Blur(out_channel),
					EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
					nn.AvgPool2d(2),

				)

		else:
			self.conv2 = nn.Sequential(
				EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),

			)
		self.bn1 = SplitBatchNorm2d(in_channel) if split else nn.BatchNorm2d(in_channel)
		self.bn2 = SplitBatchNorm2d(out_channel) if split else nn.BatchNorm2d(out_channel)
		# self.skip = EqualConv2d(in_channel, out_channel, 1, stride=2, padding=0, bias=False)
		self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1), nn.AvgPool2d(2))

	def forward(self, input, res=True):
		out = self.conv1(input)

		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if res:
			skip = self.skip(input)
			out = (skip + out ) / math.sqrt(2)
		out = self.activation(out)
		return out



class SNConvBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			padding,
			kernel_size2=None,
			padding2=None,
			downsample=False,
			fused=False,
			activation_fn='lrelu',
			n_classes=10
	):
		super().__init__()

		self.activation = get_activation(activation_fn=activation_fn)

		pad1 = padding
		pad2 = padding
		if padding2 is not None:
			pad2 = padding2

		kernel1 = kernel_size
		kernel2 = kernel_size
		if kernel_size2 is not None:
			kernel2 = kernel_size2

		self.conv1 = nn.Sequential(
			spectral_norm(nn.Conv2d(in_channel, out_channel, kernel1, padding=pad1)),
		)

		if downsample:
			if fused:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
				)

			else:
				self.conv2 = nn.Sequential(
					# Blur(out_channel),
					spectral_norm(nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2)),
					nn.AvgPool2d(2),
				)
			self.skip = nn.Sequential(EqualConv2d(in_channel, out_channel, 1), nn.AvgPool2d(2))
		else:
			self.conv2 = nn.Sequential(
				spectral_norm(nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2)),
				# nn.AvgPool2d(2)
			)

			# self.conv2 = nn.AvgPool2d(4)
			# self.conv2 = nn.MaxPool2d(4)
			self.skip = EqualConv2d(in_channel, out_channel, kernel2, padding=pad2)

		# self.gain = nn.Embedding(n_classes, in_channel)
		# self.bias = nn.Embedding(n_classes, in_channel)
		# self.skip = nn.Sequential(spectral_norm(nn.Conv2d(in_channel, out_channel, 1)), nn.AvgPool2d(2))
		# self.skip = lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
		# self.lrelu = nn.LeakyReLU(0.2)

	def forward(self, input: torch.Tensor, res=False, label=None):
		out = self.conv1(input)
		# if label is not None:
		# 	# out = (1 + self.gain(label).view(label.size(0), -1, 1, 1)) * out + self.bias(label).view(label.size(0), -1, 1, 1)
		# 	out = out + torch.sum(self.gain(label) * torch.sum(out, dim=[2, 3]), 1, keepdim=True).view(label.size(0), -1, 1, 1)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.activation(out)
		if res:
			skip = self.skip(input)
			out = (skip + out ) / math.sqrt(2)

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


class EBM(nn.Module):
	def __init__(self, base_channel=256, fused=False, spectral=False, from_rgb_activate=False, add_attention=False, res=True, projection=True,
	             activation_fn='lrelu', bn=False, split=False, num_classes=10):
		super().__init__()
		self.attention = nn.ModuleDict()

		if add_attention:
			self.attention  =nn.ModuleDict({'4': Attention(in_dim=base_channel * 2),
		                                '5': Attention(in_dim=base_channel * 2)})
		self.res = res

		kernel_size = 3
		padding = 1
		self.activation = get_activation(activation_fn=activation_fn)

		if spectral:
			self.progression = nn.ModuleDict(
				{
					'0': SNConvBlock(base_channel // 8, base_channel // 4, kernel_size, 1, downsample=True, fused=fused, activation_fn=activation_fn, n_classes=num_classes),
					# 512
					'1': SNConvBlock(base_channel // 4, base_channel // 2, kernel_size, 1, downsample=True, fused=fused, activation_fn=activation_fn, n_classes=num_classes),  # 256

					'2': SNConvBlock(base_channel // 2, base_channel, kernel_size, padding, downsample=True, fused=fused, activation_fn=activation_fn,
					                 n_classes=num_classes),  # 128
					'3': SNConvBlock(base_channel, base_channel * 2, kernel_size, padding, downsample=True, fused=fused, activation_fn=activation_fn, n_classes=num_classes),
					# 64
					'4': SNConvBlock(base_channel * 2, base_channel * 4, kernel_size, padding, downsample=True, activation_fn=activation_fn, n_classes=num_classes),  # 32
					'5': SNConvBlock(base_channel * 4, base_channel * 4, kernel_size, padding, downsample=True, activation_fn=activation_fn, n_classes=num_classes),  # 16
					'6': SNConvBlock(base_channel * 4, base_channel * 4, kernel_size, padding, downsample=True, activation_fn=activation_fn, n_classes=num_classes),  # 8
					'7': SNConvBlock(base_channel * 4, base_channel * 4, kernel_size, padding, downsample=True, activation_fn=activation_fn, n_classes=num_classes),  # 4
					'8': SNConvBlock(base_channel * 4, base_channel * 4, kernel_size, padding, 4, 0, activation_fn=activation_fn)
				}
			)
			self.final_conv = SNConvBlock(base_channel * 4, base_channel * 4, kernel_size, padding, 4, 0, activation_fn=activation_fn)
		else:
			self.progression = nn.ModuleDict(
				{
					'0': ConvBlock(base_channel // 8, base_channel//4, 3, 1, downsample=True, fused=fused, activation_fn=activation_fn),  # 512
					'1': ConvBlock(base_channel // 4, base_channel // 2, 3, 1, downsample=True, fused=fused, activation_fn=activation_fn),  # 256

					'2': ConvBlock(base_channel // 2, base_channel, 3, 1, downsample=True, fused=fused, activation_fn=activation_fn),  # 128
					'3': ConvBlock(base_channel, base_channel * 2, 3, 1, downsample=True, fused=fused, activation_fn=activation_fn),  # 64
					'4': ConvBlock(base_channel * 2, base_channel * 4, 3, 1, downsample=True, activation_fn=activation_fn),  # 32
					'5': ConvBlock(base_channel * 4, base_channel * 4, 3, 1, downsample=True, activation_fn=activation_fn),  # 16
					'6': ConvBlock(base_channel * 4, base_channel * 4, 3, 1, downsample=True, activation_fn=activation_fn),  # 8
					'7': ConvBlock(base_channel * 4, base_channel * 4, 3, 1, downsample=True, activation_fn=activation_fn),  # 4
					'8': ConvBlock(base_channel * 4, base_channel * 4, 3, 1, 4, 0, activation_fn=activation_fn)
				}
			)
			self.final_conv = ConvBlock(base_channel * 4, base_channel * 4, 3, 1, 4, 0, activation_fn=activation_fn)

		def from_rgb(out_channel):
			if from_rgb_activate:
				return nn.Sequential(EqualConv2d(1, out_channel, 1), self.activation)
			else:
				return EqualConv2d(1, out_channel, 1)

		self.ins_norm = nn.InstanceNorm2d(base_channel * 4)
		make_from_rgb = lambda x: from_rgb(x)

		self.from_rgb = nn.ModuleList(
			[
				make_from_rgb(base_channel//8),
				make_from_rgb(base_channel//4),

				make_from_rgb(base_channel//2),
				make_from_rgb(base_channel),
				make_from_rgb(base_channel*2),
				make_from_rgb(base_channel*4),
				make_from_rgb(base_channel*4),
				make_from_rgb(base_channel*4),
				make_from_rgb(base_channel*4),
			]
		)

		self.n_layer = len(self.progression)

		self.embedding = spectral_norm(nn.Embedding(num_classes, base_channel * 4))
		projection_layers = []

		if projection:
			# for _ in range(4):
			# 	projection_layers.extend([EqualLinear(base_channel * 4, base_channel*4), nn.LeakyReLU(0.2)])
			# projection_layers.append(EqualLinear(base_channel * 4, 1))
			# self.linear = nn.Sequential(*projection_layers)
			self.linear = nn.Sequential(EqualLinear(base_channel * 4, base_channel*16), self.activation, EqualLinear(base_channel * 16, 1))

		else:
			self.linear = EqualLinear(base_channel*4, 1)

		self.register_buffer('moving_latent', torch.zeros((base_channel * 4)))

	def forward(self, input, step=0, alpha=-1, label=None):
		out = None

		for i in range(step, 0, -1):
			index = self.n_layer - i - 1

			if i == step:
				out = self.from_rgb[index](input)

			out = self.progression[str(index)](out, self.res)
			# if index in self.attention.keys():
			# 	out = self.attention[str(index)](out)

			if i > 0:
				if i == step and 0 <= alpha < 1:
					# alpha = 0.5
					skip_rgb = F.avg_pool2d(input, 2)
					# skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)
					skip_rgb = self.from_rgb[index + 1](skip_rgb)

					# out = (1. - self.alpha[i]) * skip_rgb + self.alpha[i] * out
					# out = (1 - alpha) * skip_rgb + alpha * out

					out = (1 - alpha) * skip_rgb + alpha * out

		assert out is not None

		h = self.final_conv(out, res=False)
		h = torch.flatten(h, start_dim=1)

		out = self.linear(h)
		if label is not None:
			out = out + torch.sum(self.embedding(label) * h, 1, keepdim=True)
		# print(input.size(), out.size(), step)


		return out