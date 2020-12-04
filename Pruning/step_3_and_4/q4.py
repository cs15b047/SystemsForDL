import torch
import torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import mlflow
import mlflow.pytorch
import pickle
from tqdm import tqdm
import math
import numpy as np

from utils import get_datasets

num_channels_1, num_channels_2, num_neurons_1, num_neurons_2 = 12, 16, 100, 10

layer = "fc1"
# layer = "3d5ecc3a0cf54565953d8ad3ac370306"

model = mlflow.pytorch.load_model(layer)

with open(layer + ".pkl", "rb") as pkl:
	pruned_config = pickle.load(pkl)

model_config = {
	"input": [0],
	"conv1.weight": list(range(num_channels_1)),
	"conv2.weight": list(range(num_channels_2)),
	"fc1.weight": list(range(num_neurons_1)),
	"fc2.weight": list(range(num_neurons_2))
}

prev_layers = {
	"conv1.weight": "input",
	"conv2.weight": "conv1.weight",
	"fc1.weight": "conv2.weight",
	"fc2.weight": "fc1.weight",
}

# Override with pruned config
for (lay, config) in pruned_config.items():
	model_config[lay + ".weight"] = config["total_units"]

new_channels_1, new_channels_2, new_neurons_1, new_neurons_2 = len(model_config["conv1.weight"]), len(model_config["conv2.weight"]), len(model_config["fc1.weight"]), len(model_config["fc2.weight"])

print(model_config)

# Get reqd wts left after pruning using model wts and pruning config
state_dict = model.state_dict()
# print("State Dict of model")
# for k, v in state_dict.items():
# 	print(k, v.shape)
# exit(0)

# Take care of flattening separately: Reshape flattened layer as fc size x num channels x size of image
flattened_data = (state_dict["fc1.weight"].data)
reshaped_data = flattened_data.view(num_neurons_1, num_channels_2, -1)
pruned_wts_for_fc1_layer = reshaped_data[model_config["fc1.weight"]][:, model_config["conv2.weight"], :] # Index twice, first into fc neurons, then in channels as both can't be done together

# print(state_dict["conv2.weight"].data.shape)

# Pruned wts
reqd_wts = {
	"conv1.weight": (state_dict["conv1.weight"].data)[model_config["conv1.weight"]],
	"conv1.bias": (state_dict["conv1.bias"].data)[model_config["conv1.weight"]],

	"conv2.weight": (state_dict["conv2.weight"].data)[model_config["conv2.weight"]] [:, model_config["conv1.weight"]], # Double indexing into current channels and previous input channel
	"conv2.bias": (state_dict["conv2.bias"].data)[model_config["conv2.weight"]],

	"fc1.weight": pruned_wts_for_fc1_layer.view(new_neurons_1, -1), # Flatten the channel dimension
	"fc1.bias": (state_dict["fc1.bias"].data)[model_config["fc1.weight"]],

	"fc2.weight": (state_dict["fc2.weight"].data)[:, model_config["fc1.weight"]],
	"fc2.bias": (state_dict["fc2.bias"].data),
}

# reqd_wts = {
# 	"conv1.weight": (state_dict["conv1.weight"].data),
# 	"conv1.bias": (state_dict["conv1.bias"].data),

# 	"conv2.weight": (state_dict["conv2.weight"].data),
# 	"conv2.bias": (state_dict["conv2.bias"].data),
	
# 	"fc1.weight": (state_dict["fc1.weight"].data),
# 	"fc1.bias": (state_dict["fc1.bias"].data),
	
# 	"fc2.weight": (state_dict["fc2.weight"].data),
# 	"fc2.bias": (state_dict["fc2.bias"].data),
# }

reqd_wts["conv1.weight"] = reqd_wts["conv1.weight"].view(reqd_wts["conv1.weight"].shape[0], -1)
reqd_wts["conv2.weight"] = reqd_wts["conv2.weight"].view(reqd_wts["conv2.weight"].shape[0], -1)

# for (k, v) in reqd_wts.items():
# 	print(k, v.shape)

conv_wts_1 = Variable(torch.Tensor(reqd_wts["conv1.weight"]), requires_grad=True)
conv_bias_1 = Variable(torch.Tensor(reqd_wts["conv1.bias"]), requires_grad=True)

conv_wts_2 = Variable(torch.Tensor(reqd_wts["conv2.weight"]), requires_grad=True)
conv_bias_2 = Variable(torch.Tensor(reqd_wts["conv2.bias"]), requires_grad=True)

fc_1 = Variable(torch.Tensor(reqd_wts["fc1.weight"]), requires_grad=True)
fc_bias_1 = Variable(torch.Tensor(reqd_wts["fc1.bias"]), requires_grad=True)

fc_2 = Variable(torch.Tensor(reqd_wts["fc2.weight"]), requires_grad=True)
fc_bias_2 = Variable(torch.Tensor(reqd_wts["fc2.bias"]), requires_grad=True)


conv1 = nn.Conv2d(1, 12, 3);
conv1.weight.data = reqd_wts["conv1.weight"]; conv1.bias.data = reqd_wts["conv1.bias"]

conv2 = nn.Conv2d(12, 16, 5);
conv2.weight.data = reqd_wts["conv2.weight"]; conv2.bias.data = reqd_wts["conv2.bias"]

fc1 = nn.Linear(7744, 100);
fc1.weight.data = reqd_wts["fc1.weight"]; fc1.bias.data = reqd_wts["fc1.bias"]

fc2 = nn.Linear(100, 10); conv2.weight.data = reqd_wts["conv2.weight"]
fc2.weight.data = reqd_wts["fc2.weight"]; fc2.bias.data = reqd_wts["fc2.bias"]

print(conv_wts_1.size(), conv_wts_2.size(), fc_1.size(), fc_2.size(), conv_bias_1.size(), conv_bias_2.size(), fc_bias_1.size(), fc_bias_2.size())
#############^^^^^^^^^ Weights loaded from pruned model^^^^^^^^^#####################################

##############Forward pass by converting conv operation to GEMM #############
unfold_1 = nn.Unfold(kernel_size = (3, 3))
unfold_2 = nn.Unfold(kernel_size = (5, 5))

# def fwd_pass_2(x):
# 	x = F.relu(conv1(x))
# 	x = F.relu(conv2(x))
# 	x = x.view(x.size(0), -1)
# 	x = F.relu(fc1(x))
# 	x = fc2(x)
# 	return x

def conv(unfold_op, x, conv_wts, conv_bias):
	x = unfold_op(x)
	x = torch.matmul(conv_wts, x)
	x = x + (conv_bias.unsqueeze(0).unsqueeze(-1))
	x = x.view(x.size(0), x.size(1), int(math.sqrt(x.size(2))), int(math.sqrt(x.size(2))))
	return x

def fwd_pass(x):
	x = conv(unfold_1, x, conv_wts_1, conv_bias_1)
	x = F.relu(x)
	x = conv(unfold_2, x, conv_wts_2, conv_bias_2)
	x = F.relu(x)
	x = x.view(x.size(0), -1)
	x = torch.matmul(x, torch.t(fc_1)) + fc_bias_1.unsqueeze(0)
	x = F.relu(x)
	x = torch.matmul(x, torch.t(fc_2)) + fc_bias_2.unsqueeze(0)
	return x
##############################################################################
	
#################################Data fetch####################################
train_dataset, val_dataset, test_dataset = get_datasets(".")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256,
											shuffle = True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 256,
											shuffle = True, num_workers = 8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256,
											shuffle = True, num_workers = 8)
#################################################################################
# optimizer = optim.SGD([conv_wts_1, conv_wts_2, fc_1, fc_2], lr = 0.01, momentum = 0.9)
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

def inference(epoch):
	correct = 0
	train_loss = 0
	with torch.no_grad():
		for cnt, (imgs, labels) in enumerate(tqdm(train_loader), start = 1):
			logits = model(imgs)
			loss = F.cross_entropy(logits, labels)
			pred_prob = F.softmax(logits, dim = 1)

			pred_labels = np.argmax(pred_prob.data.numpy(), axis = 1)
			batch_correct = np.sum(pred_labels == labels.data.numpy())
			correct += batch_correct
			train_loss += loss.data.numpy()
			# print(batch_correct)

	train_accuracy = 100 * (correct / len(train_dataset))
	train_loss = train_loss / cnt
	print(train_accuracy, train_loss)

for epoch in range(10):
	inference(epoch)
