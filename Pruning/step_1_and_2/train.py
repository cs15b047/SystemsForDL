import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import mlflow
import mlflow.pytorch
import numpy as np
import argparse
from tqdm import tqdm

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
model = Model(model_config)
optimizer = optim.SGD(model.parameters(), lr = hyperparams.lr, momentum = hyperparams.momentum)

def train(epoch):
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
		optimizer.step()

		pred_labels = np.argmax(pred_prob.data.numpy(), axis = 1)
		correct += np.sum(pred_labels == labels.data.numpy())
		train_loss += loss.data.numpy()

	train_accuracy = 100 * (correct / len(train_dataset))
	train_loss = train_loss / cnt

	mlflow.log_metric("train_loss", train_loss, step = epoch)
	mlflow.log_metric("train_accuracy", train_accuracy, step = epoch)


def test(epoch):
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
	return (test_accuracy >= 90)

def validate(epoch):
	model.eval()
	correct = 0
	val_loss = 0
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
	return (val_accuracy >= 90)


model_path = "./"
with mlflow.start_run() as run:
	print("Run ID: %s" % (run.info.run_uuid))
	for (key, val) in vars(args).items():
		mlflow.log_param(key, val)

	for epoch in range(1, args.max_epochs + 1):
		print("Epoch: %s" % (epoch))

		train(epoch)
		done = validate(epoch)
		test(epoch)
		if done:
			break
	mlflow.pytorch.save_model(model, model_path + str(run.info.run_uuid))