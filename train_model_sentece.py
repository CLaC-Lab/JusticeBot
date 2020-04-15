"""
Training script for the binary sentence classifier
"""
import torch
import gensim
from src.models import SentenceClassifier
from src.dataset import SentenceDataset
from src.train import trainIters

device = torch.device("cpu")

embeddings_file = "../frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True, unicode_errors='ignore')
embeddings_tensor = torch.FloatTensor(embeddings.vectors)
pickle_file = "data/dataset_sentences_facts_non_facts20200311.pickle"
dataset = SentenceDataset(pickle_file=pickle_file, embeddings_file=embeddings_file, n_sentences=24)
tr = int(len(dataset)*.70)
vd = int(len(dataset)*.10)
ts = len(dataset) - tr - vd
train_ds, valid_ds, test_ds = torch.utils.data.random_split(dataset, [tr, vd, ts])
model = SentenceClassifier(embeddings_tensor=embeddings_tensor)

## CONFIG
batch_size = 8
n_epochs = 5
learning_rate = 1e-3
weight_decay = 0
clip = .2
## /CONFIG

t, v = trainIters(model, train_ds, valid_ds, batch_size, n_epochs, learning_rate, weight_decay, clip, device=device)
