import pickle
import torch
from src.dataset import FactsOrAnalysisDatasetRNN
from transformers import CamembertModel, CamembertTokenizer, CamembertForSequenceClassification
from src.models import FactOrAnalysisRNN
from tqdm import tqdm

device = torch.device("cuda")

print("Loading camemBERT model and tokeniser…")
camembert = CamembertModel.from_pretrained('camembert-base')
camembert.eval()
tokeniser = CamembertTokenizer.from_pretrained('camembert-base')
dataset = "data/dataset_docs_facts_non_facts20200311.pickle"
dataset = FactsOrAnalysisDatasetRNN(dataset, tokeniser, n_read='all')

class Reduced(torch.nn.Module):
    def __init__(self,camembert):
        super(Reduced,self).__init__()
        self.camembert = camembert
        self.reduce = torch.nn.Linear(in_features=768, out_features=64)
        
    def forward(self,x):
        x = self.camembert(x)[1]
        x = self.reduce(x)
        return x
    
class camemWRAP(torch.nn.Module):
    def __init__(self,camembert):
        super(camemWRAP, self).__init__()
        self.camembert = camembert
    
    def forward(self, x):
        return self.camembert(x)[1]

def createDataset(dim, model):
    n_doc = 0
    failed_docs = []
    print("Writing camemBERT representations to disc…")
    for doc in tqdm(dataset):
        sentences = doc[0]
        annotations = doc[1]
        with open("data/camemBERT_representations_{}/{}.pickle".format(str(dim),n_doc),"ab") as file:
            for sentence in tqdm(sentences,desc="Writing sentences"):
                try:
                    sentence = sentence.to(device)
                    sentence = model(sentence.unsqueeze(0))
                    sentence = sentence.to(torch.device('cpu'))
                    pickle.dump(sentence, file)
                except RuntimeError:
                    print("\nDocument {} produced an error. Skipping it…".format(n_doc))
                    failed_docs.append(n_doc)
                    break
            pickle.dump(annotations.to(torch.device('cpu')), file)
        n_doc += 1
    if failed_docs != []:
        print("\nUnable to add documents {} because of RuntimeError".format(failed_docs))
    print("\nDone!")

model = Reduced(camembert)
model.eval()
model.to(device)
createDataset(dim=64,model=model)
model = camemWRAP(camembert)
model.eval()
model.to(device)
createDataset(dim=768,model=model)
