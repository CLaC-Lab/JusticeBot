"""
training script for the LSTMLinear module that
takes a document represented as a series of camemBERT
outputs and outputs a binary sequence
"""
import torch
from src.dataset import DocumentDataset
from src.models import AttEncoderDecoder
from src.train import trainIters

MAX_LENGTH = 256

device = torch.device("cuda")
ds_loc = "data/camemBERT_representations_64/"
dataset = DocumentDataset(ds_loc)
tr = int(len(dataset)*.70)
vd = int(len(dataset)*.10)
ts = len(dataset) - tr - vd
train_dset, valid_dset, test_dset = torch.utils.data.random_split(dataset, [tr, vd, ts])
model = AttEncoderDecoder(hidden_size=16, max_length=MAX_LENGTH).to(device)

## CONFIG
batch_size = 1
n_epochs = 20
learning_rate = 1e-4
weight_decay = 0
clip = .2

t, v = trainIters(model, train_dset, valid_dset, batch_size, n_epochs, learning_rate, weight_decay, clip, device, collate_fn=None)

print("Done! See the output file for results.\n")
