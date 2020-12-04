import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import mlflow
import mlflow.pytorch
import numpy as np
import argparse
from tqdm import tqdm
import os
import pickle

from utils import *
from model import Model

parser = argparse.ArgumentParser()
args = parser.parse_args()

conv1_channels, conv1_kernel, conv2_channels, conv2_kernel, fc1_size = \
		int(input()), int(input()), int(input()), int(input()), int(input())
weight_init, batch_size, lr, momentum, workers, max_epochs = \
		str(input()), int(input()), float(input()), float(input()), 8, int(input())
args_dict = {
	"conv1_channels": conv1_channels, "conv1_kernel": conv1_kernel, "conv2_channels": conv2_channels, "conv2_kernel": conv2_kernel, "fc1_size": fc1_size,
	"weight_init": weight_init, "batch_size": batch_size, "lr": lr, "momentum": momentum,
	"workers": workers, "max_epochs": max_epochs
}
args.__dict__ = args_dict

model_config = {
	"conv1_channels": args.conv1_channels,
	"conv1_kernel": args.conv1_kernel,
	"conv2_channels": args.conv2_channels,
	"conv2_kernel": args.conv2_kernel,
	"fc1_size": args.fc1_size,
	"weight_init": get_wt_init_fn(args.weight_init)
}

hyperparams = Params(args.batch_size, args.lr, args.momentum)

train_dataset, val_dataset, test_dataset = get_datasets(".")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = hyperparams.batch_size,
											shuffle = True, num_workers = args.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = hyperparams.batch_size,
											shuffle = True, num_workers = args.workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = hyperparams.batch_size,
											shuffle = True, num_workers = args.workers)
# global model
# model = Model(model_config)
# optimizer = optim.SGD(model.parameters(), lr = hyperparams.lr, momentum = hyperparams.momentum)

# def train(epoch, prune = False, layer = None, channels = None, pruned_model = None):
def train(epoch, prune = False, pruning_info = None, pruned_model = None):
	if prune:
		model = pruned_model

	model.train()
	correct = 0
	train_loss = 0
	for cnt, (imgs, labels) in enumerate(tqdm(train_loader), start = 1):
		imgs, labels = Variable(imgs), Variable(labels)
		logits = model(imgs)
		loss = F.cross_entropy(logits, labels)
		pred_prob = F.softmax(logits, dim = 1)

		optimizer.zero_grad()
		loss.backward()
		
		if prune:
			for idx, (name, p) in enumerate(model.named_parameters()):
				# if layer == name:
				name_stripped = name.split(".")[0] # Required for layer and its bias term
				if name_stripped in pruning_info:
					# print(name, pruning_info[name], pruning_info[name]["pruned_units"])
					p.grad[pruning_info[name_stripped]["pruned_units"]]= 0
					# p.grad[channels] = 0

		optimizer.step()

		pred_labels = np.argmax(pred_prob.data.numpy(), axis = 1)
		correct += np.sum(pred_labels == labels.data.numpy())
		train_loss += loss.data.numpy()

	train_accuracy = 100 * (correct / len(train_dataset))
	train_loss = train_loss / cnt

	mlflow.log_metric("train_loss", train_loss, step = epoch)
	mlflow.log_metric("train_accuracy", train_accuracy, step = epoch)
	return train_accuracy


def test(epoch, prune = False, pruned_model = None):
	if prune:
		model = pruned_model
	model.eval()
	correct = 0
	test_loss = 0
	with torch.no_grad():
		for imgs, labels in tqdm(test_loader):
			imgs, labels = Variable(imgs), Variable(labels)
			logits = model(imgs)
			loss = F.cross_entropy(logits, labels, reduction = 'sum')
			pred_prob = F.softmax(logits, dim = 1)

			pred_labels = np.argmax(pred_prob.data.numpy(), axis = 1)
			correct += np.sum(pred_labels == labels.data.numpy())
			test_loss += loss.data.numpy()

	test_loss = test_loss / len(test_dataset)
	test_accuracy = 100 * (correct / len(test_dataset))

	mlflow.log_metric("test_loss", test_loss, step = epoch)
	mlflow.log_metric("test_accuracy", test_accuracy, step = epoch)
	return test_accuracy, (test_accuracy >= 90)

def validate(epoch, prune = False, pruned_model = None):
	if prune:
		model = pruned_model
	model.eval()
	correct = 0
	val_loss = 0
	# print(model.conv1.weight)
	# print("valalalalalaalal")

	with torch.no_grad():
		for imgs, labels in tqdm(val_loader):
			imgs, labels = Variable(imgs), Variable(labels)
			logits = model(imgs)
			loss = F.cross_entropy(logits, labels, reduction = 'sum')
			pred_prob = F.softmax(logits, dim = 1)

			pred_labels = np.argmax(pred_prob.data.numpy(), axis = 1)
			correct += np.sum(pred_labels == labels.data.numpy())
			val_loss += loss.data.numpy()

	val_loss = val_loss / len(test_dataset)
	val_accuracy = 100 * (correct / len(test_dataset))

	mlflow.log_metric("val_loss", val_loss, step = epoch)
	mlflow.log_metric("val_accuracy", val_accuracy, step = epoch)
	return val_accuracy, (val_accuracy >= 90)

def get_layer_and_idx(layer_name, model):
	if layer_name == "conv1":
		pp = model.conv1.weight
	elif layer_name == "conv2":
		pp = model.conv2.weight
	elif layer_name == "fc1":
		pp = model.fc1.weight
	elif layer_name == "fc2":
		pp = model.fc2.weight

	for idx, p in enumerate(model.parameters()):
		if(p.data.shape == pp.data.shape and torch.all(p.eq(pp))):
			return pp.data, idx
	return None, None

def get_channel_to_prune(layer, total_units):
	tensor = layer.view(layer.shape[0], -1)
	norms = torch.norm(tensor, p = 2, dim = 1)
	mag, index = torch.min(norms), torch.argmin(norms)
	mag, index = mag.data.item(), index.data.item()

	# Only remaining channels are included, so indexing distorts.. So, indexing into the remaining units
	return mag, total_units[index]

def get_model_file_pth(layer_name, base_model):
	if layer_name == "conv1":
		prev_layer = base_model
	elif layer_name == "conv2":
		prev_layer = "conv1"
	elif layer_name == "fc1":
		prev_layer = "conv2"
	elif layer_name == "fc2":
		prev_layer = "fc1"

	return prev_layer

def get_pruning_info(layer_name):
	if layer_name == "conv1":
		return {}
	else:
		prev_layer = get_model_file_pth(layer_name, "")
		with open(prev_layer + ".pkl", "rb") as pkl:
			info = pickle.load(pkl)
		return info

def prune_layer(layer, layer_bias, pruning_info):
	layer[pruning_info[pruned_layer_name]["pruned_units"]] = 0
	layer_bias[pruning_info[pruned_layer_name]["pruned_units"]] = 0
	return layer, layer_bias

def update_info(layer_pruning_info, unit, ):
	layer_pruning_info["pruned_units"] += [unit]
	layer_pruning_info["pruned_units"].sort()
	layer_pruning_info["total_units"].remove(unit)
	return layer_pruning_info

training, prune = int(input()), int(input())

if training == 1:
	model_path = "./"
	with mlflow.start_run() as run:
		print("Run ID: %s" % (run.info.run_uuid))
		for (key, val) in vars(args).items():
			mlflow.log_param(key, val)

		for epoch in range(1, args.max_epochs + 1):
			print("Epoch: %s" % (epoch))

			train_accuracy = train(epoch)
			test_acc, done = validate(epoch)
			test(epoch)
			if done:
				break
		mlflow.pytorch.save_model(model, model_path + str(run.info.run_uuid))
elif prune == 1:

	# Load previous pruned model and pruning information regarding pruned channels
	run_uuid, pruned_layer_name = str(input()), str(input())

	# pruned_layer_name = pruned_layer_name + "." + "weight"
	model_pth = os.path.join(get_model_file_pth(pruned_layer_name, run_uuid))
	model = mlflow.pytorch.load_model(model_pth)

	state_dict = model.state_dict()
	layer, layer_bias = state_dict[pruned_layer_name + ".weight"].data, state_dict[pruned_layer_name + ".bias"].data
	num_units = int(layer.shape[0])

	# Dict of all prev layers and their channels to prune: {"Layer_Name" --> {"pruned": [], "remn": []}}
	pruning_info = get_pruning_info(pruned_layer_name)
	print(pruning_info)

	################## Initial evaluation of model###################
	val_acc, _ = validate(0, prune, model)
	mlflow.log_metric("Pruning accuracy using layer {}".format(pruned_layer_name), val_acc)
	print("Init val acc: {}".format(val_acc))
	##############################################################
	
	############# Init########
	pruned_units = []
	total_units = list(range(num_units))
	pruning_info[pruned_layer_name] = {"pruned_units": [], "total_units": list(range(num_units))}

	optimizer = optim.SGD(model.parameters(), lr = hyperparams.lr, momentum = hyperparams.momentum)

	######## Prune in loop, till accuracy doesn't fall by more than 1% in single step#########
	channel_num = 0 # number of channels pruned
	while(len(pruned_units) < num_units - 1):
		
		#Heuristic: Norm of the dropped channel/neuron is lowest
		# Exclude previously zeroed channels
		layer, layer_bias = state_dict[pruned_layer_name + ".weight"].data, state_dict[pruned_layer_name + ".bias"].data
		mag, unit = get_channel_to_prune(layer[total_units], total_units)
		pruning_info[pruned_layer_name] = update_info(pruning_info[pruned_layer_name], unit)
		pruned_units += [unit]
		pruned_units.sort()
		total_units.remove(unit)

		# Drop unit: previously pruned layers already 0
		layer, layer_bias = prune_layer(layer, layer_bias, pruning_info)
		state_dict[pruned_layer_name + ".weight"].copy_(layer)
		state_dict[pruned_layer_name + ".bias"].copy_(layer_bias)



		val_acc_pruned, _ = validate(channel_num, prune, model)
		mlflow.log_metric("Pruning accuracy using layer {}".format(pruned_layer_name), val_acc_pruned)
		print("Pruned val acc: %f"%(val_acc_pruned))
		test_acc, _ = test(-1, prune, model)
		print("Pruned test accuracy %f"%(test_acc))
		
		# Retrain and check
		val_acc_retrained = val_acc_pruned
		# if val_acc - val_acc_pruned > 1:
		for epoch in range(1):
			# train(epoch, prune, pruned_layer_name, pruned_units, model)
			train(epoch, prune, pruning_info, model)
			val_acc_retrained, _ = validate(epoch, prune, model)
			print("Retrained val acc:%f"%(val_acc_retrained))
			mlflow.log_metric("Pruning accuracy using layer {}".format(pruned_layer_name), val_acc_retrained)
			test_acc, _ = test(epoch, prune, model)
			print("Retrained test acc: %f"%(test_acc))
		# print(state_dict[pruned_layer_name])
		print("Pruned channels: {}".format(pruned_units))
		val_acc_retrained, _ = validate(5, prune, model)

		print("Previous epoch val acc: {}, Pruned val acc: {}, Retrained val acc: {}".format(val_acc, val_acc_pruned, val_acc_retrained))

		# Measuring difference in accuracy across 1 channel pruning

		# Don't prune if drop is more than 1%
		if (val_acc - val_acc_retrained > 1) or (test_acc <= 87):
			break
		else:
			val_acc = val_acc_retrained

	print(pruned_units, total_units)	
	model_path = "./"
	run = mlflow.active_run()
	mlflow.pytorch.save_model(model, model_path + str(pruned_layer_name))
	with open(pruned_layer_name + ".pkl", "wb") as pkl:
		pickle.dump(pruning_info, pkl)
