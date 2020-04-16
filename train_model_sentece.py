"""
Training and evaluation script for the binary sentence classifier
"""
import torch
import gensim
import numpy as np
from src.models import SentenceClassifier
from src.dataset import SentenceDataset
from src.train import trainIters, test
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings_file = "../frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True, unicode_errors='ignore')
embeddings_tensor = torch.FloatTensor(embeddings.vectors)
pickle_file = "data/dataset_sentences_facts_non_facts20200311.pickle"
dataset = SentenceDataset(pickle_file=pickle_file, embeddings_file=embeddings_file, n_sentences=16)
tr = int(len(dataset)*.70)
vd = int(len(dataset)*.10)
ts = len(dataset) - tr - vd
train_ds, valid_ds, test_ds = torch.utils.data.random_split(dataset, [tr, vd, ts])
model = SentenceClassifier(embeddings_tensor=embeddings_tensor).to(device)

## CONFIG
batch_size = 8
n_epochs = 1
learning_rate = 1e-4
weight_decay = 0
clip = .2
## /CONFIG

t, v = trainIters(model, 
    train_ds, 
    valid_ds, 
    batch_size, 
    n_epochs, 
    learning_rate, 
    weight_decay, 
    clip, 
    device=device, 
    collate_fn="sentence")

torch.save(model.state_dict(), "sentence_clf.pt")
model.eval()
pred = []
targ = []
for sentence, annotation in test_ds:
    length = torch.tensor([len(sentence)]).to(device)
    pred.append(model(sentence.unsqueeze(0), length))
    targ.append(annotation)
pred = torch.tensor(pred).unsqueeze(0)
targ = torch.tensor(targ).unsqueeze(0)
acc, prec, rec = test(targ, pred)
with open("sentence_clf.eval", "w") as file:
    file.write("accuracy\tprecision\trecall\n")
    file.write("{:.4f}\t\t{:.4f}\t\t{:.4f}".format(acc, prec, rec))