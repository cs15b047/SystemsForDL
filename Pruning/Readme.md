#Hyperparameter grid search
Run "python gen_hyperparams.py" with appropriate hyperparameter ranges to form a grid of hyperparams.
Run python monitor_train.py with appropriate names of directory of hyperpaprameter input files provided in the code.

#Pruning
a. Run 1. python prune.py < demo_prune_ip_1 , 2. python prune.py < demo_prune_ip_2, 3. python prune.py < demo_prune_ip_3 in the same order to prune model in conv1, conv2 and fc1 layers respectively.
b. Run python q4.py by setting the appropriate layer variable: "conv1", "conv2", "fc1", which instructs to load model pruned in conv1, (conv1 + conv2) and (conv1 + conv2 + fc1) layers respectively.
