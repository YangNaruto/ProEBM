from PIL import Image
import torchvision as tv, torchvision.transforms as tr
import os
from utils import dataset_util
from torch.utils.data import DataLoader

transform = tr.Compose(
	[
		# tr.Resize(32),
		# tr.RandomHorizontalFlip(),
		tr.ToTensor(),
		tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
	]
)

dataset_path = os.path.join('./data', 'celeba-c')
dataset = dataset_util.MultiResolutionDataset(dataset_path, transform)

for resolution in [8, 16, 32]:
	dataset.resolution = resolution
	loader = DataLoader(dataset, shuffle=False, batch_size=4, num_workers=1, drop_last=True, pin_memory=True)
	data_loader = iter(loader)
	real_image, real_label = next(data_loader)

	tv.utils.save_image(real_image, f'{resolution}.png', padding=0, normalize=True, range=(-1., 1.), nrow=2)
