import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from src.dataset import FactsOrAnalysisDS_BERT
from src.models import FoaCamemBERTlinear
from src.train import trainIters
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print("Loading camemBERT models…")
device = torch.device("cuda")
camembert_seq = CamembertForSequenceClassification.from_pretrained('camembert-base',num_labels=1)
tokeniser = CamembertTokenizer.from_pretrained('camembert-base')
print("Loading dataset…")
ds = "data/dataset_sentences_facts_non_facts20200311.pickle"
ds = FactsOrAnalysisDS_BERT(ds,tokeniser,n_read="all")
tr = int(len(ds)*.70)+1
vd = int(len(ds)*.10)+1
ts = len(ds) - tr - vd
print("Splitting dataset into training, validation and test sets…")
train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds,[tr,vd,ts])

model=FoaCamemBERTlinear(camembert_seq).to(device)

## CONFIG
batch_size=4
n_epochs=5
learning_rate=1e-4
weight_decay=0
clip=.2
## /CONFIG
print("Training…")
t,v=trainIters(model, train_ds, valid_ds, batch_size, n_epochs, learning_rate, weight_decay, clip, device)
print("Training complete!")
