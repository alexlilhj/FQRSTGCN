import torch

INPUT_DIM = 2
local_epoch = 10
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_owners = 3
communication_num = 100
batch_size = 128
quantile_list_num = 11
input_length = 10

