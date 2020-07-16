import torch


class EMA(object):
	def __init__(self, source, target, decay=0.9999, start_itr=0):
		self.source = source
		self.target = target
		self.decay = decay
		# Optional parameter indicating what iteration to start the decay at
		# Initialize target's params to be source's
		self.source_dict = self.source.state_dict()
		self.target_dict = self.target.state_dict()
		print('Initializing EMA parameters to be source parameters...')
		with torch.no_grad():
			for key in self.source_dict:
				self.target_dict[key].data.copy_(self.source_dict[key].data)
				# target_dict[key].data = source_dict[key].data # Doesn't work!

	def update(self, decay=0.0):

		with torch.no_grad():
			for key in self.source_dict:
				self.target_dict[key].data.copy_(self.target_dict[key].data * decay
				                                 + self.source_dict[key].data * (1 - decay))
