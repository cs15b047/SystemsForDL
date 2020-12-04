import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import get_wt_init_fn

class Model(nn.Module):
	def __init__(self, config):
		# Config will contain:
		#1. Conv1: Channels, kernel size, 2. Conv2: Channels, kernel size, 3. Linear1: num hidden units, 4. Weight init method
		img_sz, img_channels, num_outputs = 28, 1, 10
		conv1_channels, conv1_kernel, conv2_channels, conv2_kernel, fc1_size, weight_init_fn = \
			config["conv1_channels"], config["conv1_kernel"], config["conv2_channels"], config["conv2_kernel"], config["fc1_size"], config["weight_init"]

		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(img_channels, conv1_channels, kernel_size=conv1_kernel)
		self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=conv2_kernel)

		new_sz = (img_sz - (conv1_kernel - 1) - (conv2_kernel - 1))
		flattened_sz = new_sz * new_sz * conv2_channels
		
		self.fc1 = nn.Linear(flattened_sz, fc1_size)
		self.fc2 = nn.Linear(fc1_size, num_outputs)

		# TODO: Check if function can be passed as an object and generates randomness
		weight_init_fn(self.conv1.weight)
		weight_init_fn(self.conv2.weight)
		weight_init_fn(self.fc1.weight)
		weight_init_fn(self.fc2.weight)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		return x

if __name__ == '__main__':
	model_config = {
		"conv1_channels": 16,
		"conv1_kernel": 3,
		"conv2_channels": 16,
		"conv2_kernel": 3,
		"fc1_size": 100,
		"weight_init": get_wt_init_fn("xavier_uniform"),
	}
	m = Model(model_config)
	x = np.random.normal(size = (1, 1, 28, 28))
	x = Variable(torch.Tensor(x))
	y = m(x)
	print(y.size())