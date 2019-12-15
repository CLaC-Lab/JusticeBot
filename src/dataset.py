import gensim,torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
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