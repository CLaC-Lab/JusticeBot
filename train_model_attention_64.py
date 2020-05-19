"""
training script for the LSTMLinear module that
takes a document represented as a series of camemBERT
outputs and outputs a binary sequence
"""
import torch
from src.dataset import DocumentDataset
from src.models import AttEncoderDecoder
from src.train import trainIters, evaluateModel

MAX_LENGTH = 256

with open("status", "w") as file:
    file.write("I AM ALIVE HERE\n")
    cudastatus = "I HAVE CUDA" if torch.cuda.is_available() else "I HAVE NO CUDA"
    file.write(cudastatus)
device = torch.device("cuda")
ds_loc = "data/camemBERT_representations_64/"
dataset = DocumentDataset(ds_loc, ds_len=None, max_length=MAX_LENGTH)
tr = int(len(dataset)*.70)
vd = int(len(dataset)*.10)
ts = len(dataset) - tr - vd
train_dset, valid_dset, test_dset = torch.utils.data.random_split(dataset, [tr, vd, ts])
with open("status", "a") as file:
    file.write("\nCREATING THE MODEL")
model = AttEncoderDecoder(input_size=64, hidden_size=512, max_len=MAX_LENGTH, device=device).to(device)

## CONFIG
batch_size = 1
n_epochs = 15
learning_rate = 1e-4
weight_decay = 0
clip = .2
with open("status", "a") as file:
    file.write("\nTRAINING")

t, v = trainIters(
    model, 
    train_dset, 
    valid_dset, 
    batch_size, 
    n_epochs, 
    learning_rate, 
    weight_decay, 
    clip, 
    device,
    input_size=64,
    lr_step=[10,13])

torch.save(model.state_dict(), "doc_segmentation.pt")
model.eval()
evaluateModel(model=model, 
    filename="doc_segmentation", 
    test_set=test_dset, 
    batch_size=batch_size,
    device=device,
    input_size=64)
# avg_acc = []
# avg_prec = []
# avg_rec = []
# avg_f1 = []
# data_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
# data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
# file = open("doc_segmentation.eval", "w")
# file.write("accuracy\tprecision\trecall\t\tf1\n")
# for sentence, annotation, lengths in data_loader:
#     prediction = model(sentence.to(device), lengths)
#     acc, prec, rec, f1 = test(annotation, prediction)
#     avg_acc.append(acc)
#     avg_prec.append(prec)
#     avg_rec.append(rec)
#     avg_f1.append(f1)
# acc = sum(avg_rec)/len(avg_rec)
# prec = sum(avg_prec)/len(avg_prec)
# rec = sum(avg_rec)/len(avg_rec)
# f1 = sum(avg_f1)/len(avg_f1)
# file.write("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(acc, prec, rec, f1))
# file.close()
# print("Done! See the output file for results.\n")
