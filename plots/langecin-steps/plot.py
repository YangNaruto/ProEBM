import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
sns.set_style('whitegrid')

stats   = {}
for step in range(30, 110, 10):
	stats[step] = []
	with open(f'log_{step}.txt') as f:
		lines  = f.readlines()
		for line in lines:
			if 'Itr' in line and 'FID' in line:
				stat = re.sub('[^A-Za-z0-9.]+', ' ', line).split(' ')
				stats[step].append([int(stat[1]), float(stat[-2])])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for steps, val in stats.items():
	xy = np.array(val)
	itr, fid = xy[:, 0], xy[:, 1]
	print(steps, np.min(fid))
	plt.plot(itr, fid, label=str(steps))

ax.set_ylim(20, 100)
plt.legend()
plt.show()
print('hello')
