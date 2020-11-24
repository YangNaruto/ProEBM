import torch
import numpy  as np
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as tf
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

plt.rcParams.update({'font.size': 15})


class Swish(torch.autograd.Function):
	@staticmethod
	def forward(ctx, i):
		result = i * torch.sigmoid(i) / 1.
		ctx.save_for_backward(i)
		return result

	@staticmethod
	def backward(ctx, grad_output):
		i = ctx.saved_variables[0]
		sigmoid_i = torch.sigmoid(i)
		return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)) / 1.)


class SwishModule(nn.Module):
	def forward(self, input_tensor):
		return Swish.apply(input_tensor)


x = torch.arange(-5, 5, 0.01)
x.requires_grad_(True)

y_relu = tf.relu(x)
y_relu_g = torch.autograd.grad(y_relu.mean(),  [x], retain_graph=False)[0].numpy()

y_lrelu = tf.leaky_relu(x)
y_lrelu_g = torch.autograd.grad(y_lrelu.mean(),  [x], retain_graph=False)[0].numpy()

y_celu = tf.celu(x)
y_celu_g = torch.autograd.grad(y_celu.mean(),  [x], retain_graph=False)[0].numpy()

y_swish = SwishModule()(x)
y_swish_g = torch.autograd.grad(y_swish.mean(),  [x], retain_graph=False)[0].numpy()

y_gelu = tf.gelu(x)
y_gelu_g = torch.autograd.grad(y_gelu.mean(),  [x], retain_graph=False)[0].numpy()


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
x = x.detach().numpy()
n = len(x)

ax[0].plot(x, y_relu.detach().numpy(), label='ReLU')
ax[0].plot(x, y_lrelu.detach().numpy(), label='LeakyReLU')
ax[0].plot(x, y_celu.detach().numpy(), label='CELU')
ax[0].plot(x, y_gelu.detach().numpy(), label='GELU')
ax[0].plot(x, y_swish.detach().numpy(), label='Swish')
ax[0].title.set_text('Activations')

ax[1].plot(x, y_relu_g*n, label='ReLU')
ax[1].plot(x, y_lrelu_g*n, label='LeakyReLU')
ax[1].plot(x, y_celu_g*n, label='CELU')
ax[1].plot(x, y_gelu_g*n, label='GELU')
ax[1].plot(x, y_swish_g*n, label='Swish')
ax[1].title.set_text('Derivatives')

ax[0].set_ylim(-1.1, 4)
ax[0].legend()
ax[1].legend()
plt.savefig('activation.png', dpi=300)
plt.show()


