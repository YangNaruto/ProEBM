import numpy as np
import torch
import torchvision as tv, torchvision.transforms as tr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('white')
import math
a2b = np.load('B2A.npy')[1:2]

bs, steps, ch, h, w = a2b.shape
print(f'steps: {steps}, bs: {bs}, size: {h, w}')


residual = np.zeros((bs, ch, h, w))
for step in range(steps - 1):
	residual += np.abs(a2b[:, step+1] - a2b[:, step])

residual = np.mean(residual, axis=1)

n_rows = int(math.sqrt(bs))
n_cols = int(math.sqrt(bs))

residual = np.reshape(residual, (n_rows, n_cols, w, h))
residual = np.transpose(residual, axes=(0, 2, 1, 3))
residual = np.reshape(residual, (n_rows*w, n_cols*h))

fig, ax = plt.subplots(1, 1, figsize=(8,8))
plt.axis('off')

ax.set_xticklabels('')
ax.set_yticklabels('')
heat = sns.heatmap(residual, cmap="YlGnBu", cbar=False)
figure = heat.get_figure()

figure.savefig('svm_conf.png', dpi=400)

# fig, ax = plt.subplots(1, 1)
# plt.axis('off')
#
# ax.set_xticklabels('')
# ax.set_yticklabels('')
# plt.imshow(residual)
#
# plt.savefig('heatmap.png', dpi=300)
