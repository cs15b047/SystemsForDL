conv1_channels = [12] #[8, 12, 16]
conv1_kernel = [3] #[3, 5]
conv2_channels = [16] #[8, 12, 16]
conv2_kernel = [5] #[3, 5]
fc1_size = [100] #[50, 100]

weight_init = ["xavier_uniform", "xavier_normal", "normal"]
batch_size = [64, 128, 256, 512]
lr = [0.1, 0.01, 0.001]
momentum = [0.9]

max_epochs = [50]

from itertools import product

prod = product(conv1_channels, conv1_kernel, conv2_channels, conv2_kernel, fc1_size, weight_init, batch_size, lr, momentum, max_epochs)
prod2 = product(conv1_channels, conv1_kernel, conv2_channels, conv2_kernel, fc1_size, weight_init, batch_size, [0.01], [0.99], max_epochs)

prod, prod2 = list(prod), list(prod2)
prod.extend(prod2)
combns = prod

import os
os.chdir("./drive/My Drive/A3/")
if not os.path.exists("train_hyps"):
	os.makedirs("train_hyps")

for idx, hyp in enumerate(combns):
	hyp = list(hyp)
	with open("train_hyps/train_hyp_{}.txt".format(idx), 'w') as f:
		for p in hyp:
			f.write("{}\n".format(str(p)))