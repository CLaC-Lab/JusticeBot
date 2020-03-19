import pickle
from src.dataset import FactsOrAnalysisDatasetRNN
from transformers import CamembertModel, CamembertTokenizer, CamembertForSequenceClassification
from src.models import FactOrAnalysisRNN
from tqdm import tqdm

print("Loading camemBERT model and tokeniser…")
camembert = CamembertModel.from_pretrained('camembert-base')
camembert.eval()
tokeniser = CamembertTokenizer.from_pretrained('camembert-base')
dataset = "data/dataset_docs_facts_non_facts20200311.pickle"
dataset = FactsOrAnalysisDatasetRNN(dataset, tokeniser, n_read="all")
# model = FactOrAnalysisRNN(camembert,hidden_size=64)

import torch
reduction = torch.nn.Linear(in_features=768, out_features=64)

class Reduced(torch.nn.Module):
    def __init__(self,camembert):
        super(Reduced,self).__init__()
        self.camembert = camembert
        self.reduce = torch.nn.Linear(in_features=768, out_features=64)
        
    def forward(self,x):
        x = self.camembert(x)[1]
        x = self.reduce(x)
        return x

model = Reduced(camembert)

n_doc = 0
failed_docs = []
print("Writing camemBERT representations to disc…")
for doc in tqdm(dataset):
    sentences = doc[0]
    with open("data/camemBERT_representations/{}.pickle".format(n_doc),"ab") as file:
    	for sentence in tqdm(sentences,desc="Writing sentences"):
            try:
                sentence = model(sentence.unsqueeze(0))
                pickle.dump(sentence, file)
            except RuntimeError:
                print("Document {} produced an error. Skipping it…".format(n_doc))
                failed_docs.append(n_doc)
    n_doc += 1
print("Unable to add documents {} because of RuntimeError".format(failed_docs))
print("\nDone!")
