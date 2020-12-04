import os

base_dir = "./drive/My Drive/A3/"
os.chdir(base_dir)
idx_file = os.path.join("file_idx.txt")

if not os.path.exists(idx_file):
	with open(idx_file, "w") as f:
		f.write("0\n")

with open(idx_file, "r") as f:
	file_idx = int((f.readlines())[0])

max_idx = 48

script = os.path.join("train.py")

for idx in range(file_idx, max_idx):
	input_file = os.path.join("train_hyps", "train_hyp_{}.txt".format(idx))
	ret_val = os.system("bash sc.sh {} {}".format(script, input_file))
	if(ret_val == 0):
		with open(idx_file, "w") as f:
			f.write("{}\n".format(idx + 1))
	else:
		print(ret_val)
		exit(0)