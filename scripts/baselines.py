import torch
from src.dataset import DocumentDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
acc = []
pre = []
rec = []
f1 = []
dataset = DocumentDataset(ds_loc="/home/andres/JusticeBot/data/camemBERT_representations_64/")
for index in tqdm(range(len(dataset)), desc="Retrieving baseline metrics"):
    doc = dataset[index][1]
    prediction = len(doc)//2
    prediction = torch.ones(prediction)
    padding = len(doc) - len(prediction)
    prediction = torch.nn.functional.pad(prediction, pad=(0, padding))
    acc.append(accuracy_score(doc, prediction))
    pre.append(precision_score(doc, prediction))
    rec.append(recall_score(doc, prediction))
    f1.append(f1_score(doc, prediction))
acc = sum(acc)/len(acc)
pre = sum(pre)/len(pre)
rec = sum(rec)/len(rec)
f1 = sum(f1)/len(f1)
print("acc: {}".format(acc))
print("pre: {}".format(pre))
print("rec: {}".format(rec))
print("f1: {}".format(f1))

# Retrieving baseline metrics: 100%|██████████| 10360/10360 [01:00<00:00, 171.18it/s]

# acc: 0.8437448789118575
# pre: 0.8588532563384854
# rec: 0.8789491190115489
# f1: 0.8310572742383662

