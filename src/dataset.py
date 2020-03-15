import torch,pickle#, gensim
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

class Lexicon:
    """Helper function. Collects vocabulary of embeddings model into key-value and value-key pairs"""
    def __init__(self,embeddings):
        self.word2index = {token: token_index for token_index, token in enumerate(embeddings.index2word)}
        self.index2word = embeddings.index2word
        self.n_words = len(embeddings.index2word)
        
def tensorFromSentence(lexicon,sentence):
    """Returns a PyTorch tensor object from a string
    
    In: embeddings model and a sentence string
    Out: long tensor object
    
    """
    index_list = []
    sentence = sentence.split(' ')
    for word in sentence:
        try:
            index = lexicon.word2index[word]
        except:
            index = 0
        index_list.append(index)
    return torch.tensor(index_list, dtype=torch.long)

class FactsOrAnalysisDataset(Dataset):
    """PyTorch Dataset class. Returns input-target tensor pairs"""
    def __init__(self,csv_file,max_length=50,embeddings_file="embeddings.bin",n_sentences=0):
        self.facts_or_analysis_frame = pd.read_csv(csv_file)
        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file,binary=True,unicode_errors='ignore')
        if n_sentences > 0:
            self.facts_or_analysis_frame = self.facts_or_analysis_frame.iloc[0:n_sentences]
        self.max_length = max_length
        self.lexicon=Lexicon(self.embeddings)
    
    def __len__(self):
        return len(self.facts_or_analysis_frame)
    
    def __getitem__(self,index):
        sentence = self.facts_or_analysis_frame.iloc[index,0]
        annotation = self.facts_or_analysis_frame.iloc[index,1]
        sentence = tensorFromSentence(self.lexicon,sentence)
        padding = self.max_length - len(sentence)
        sentence = F.pad(sentence, pad=(0, padding), mode='constant', value=0)
        annotation = torch.tensor([1,0], dtype=torch.float) if annotation==1 else torch.tensor([0,1], dtype=torch.float)
        return sentence,annotation
    
class FactsOrAnalysisDS_BERT(Dataset):
    """
    PyTorch Dataset class. Returns input-target tensor pairs
    """
    def __init__(self, pickle_file, tokeniser, n_read='all'):
        """
        IN: csv file location, BertTokeniser object
        """
        with open(pickle_file,"rb") as file:
            dataset = pickle.load(file)

        if n_read == 'all':
            self.dataset = [[torch.tensor(tokeniser.encode(datum[0])),
                           int(datum[1])] for datum in dataset]
        else:
            self.dataset = []
            for i in range(n_read):
                self.dataset.append([torch.tensor(tokeniser.encode(dataset[i][0])),
                           int(dataset[i][1])])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

class FactsOrAnalysisDatasetRNN(Dataset):
    """
    Pytorch Dataset class. Returns input-target tensor pairs
    """
    def __init__(self, pickle_file, tokeniser, n_read='all'):
        """
        IN: pickle file containing full documents, camemBERT tokeniser object
        """
        self.dataset = []
        with open(pickle_file, "rb") as file:
            dataset = pickle.load(file)

        if n_read == 'all':
            n = len(dataset)
        else:
            n = n_read

        for i in range(n):
            facts, non_facts = dataset[i]['facts'], dataset[i]['non_facts']
            facts = [tokeniser.encode(sentence) for sentence in facts]
            facts = [torch.tensor(sentence) for sentence in facts]
            non_facts = [tokeniser.encode(sentence) for sentence in non_facts]
            non_facts = [torch.tensor(sentence) for sentence in non_facts]
            #############################################
            ## TO DO: Think moar about catting tensors ##
            #############################################
            ones, zeros = len(dataset[i]['facts']), len((dataset[i]['non_facts']))
            ones, zeros = torch.ones(ones), torch.zeros(zeros)
            self.dataset.append([facts + non_facts, torch.cat([ones, zeros])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
