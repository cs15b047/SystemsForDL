import torch
import numpy as np
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import torch.nn as nn
import sys

class Params(object):
	def __init__(self, batch_size, lr, momentum):
		self.batch_size = batch_size
		self.lr = lr
		self.momentum = momentum

def get_datasets(pth):
	train_data = FashionMNIST(pth, train = True, download = True, transform = transforms.ToTensor())
	test_data = FashionMNIST(pth, train = False, download = True, transform = transforms.ToTensor())
	train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])
	return train_data, val_data, test_data

def get_wt_init_fn(fn_name):
	if(fn_name == "xavier_uniform"):
		return nn.init.xavier_uniform_
	elif(fn_name == "xavier_normal"):
		return nn.init.xavier_normal_
	elif(fn_name == "normal"):
		return nn.init.normal_
	else:
		sys.exit("Wrong weight initializer!!")
		return None

if __name__ == "__main__":
	train_data, test_data = get_datasets("./")
	print("main", len(train_data), len(test_data))