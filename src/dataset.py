import torch, pickle, os, gensim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

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

class SentenceDataset(Dataset):
    """
    Creates PyTorch dataset consisting of sentence-tag pairs 
    represented as tensors
    """
    def __init__(self, pickle_file, max_len=50, embeddings_file="embeddings.bin", n_sentences=0):
        with open(pickle_file, "rb") as file:
            self.data = pickle.load(file)
        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True, unicode_errors='ignore')
        if n_sentences > 0:
            self.data = self.data[:n_sentences]
        self.lexicon = Lexicon(self.embeddings)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data[index][0]
        annotation = self.data[index][1]
        sentence = tensorFromSentence(self.lexicon, sentence)
        padding = self.max_len - len(sentence)
        sentence = F.pad(sentence, pad=(0, padding), mode='constant', value=0)
        annotation = torch.tensor(annotation, dtype=torch.float)
        return sentence, annotation
    
class FactsOrAnalysisDS_BERT(Dataset):
    """
    PyTorch Dataset class. Returns input-target tensor pairs
    """
    def __init__(self, pickle_file, tokeniser, n_read='all'):
        """
        IN: csv file location, BertTokeniser object
        """
        self.dataset = []
        with open(pickle_file,"rb") as file:
            dataset = pickle.load(file)

        if n_read == 'all':
            n = len(dataset)
        else:
            n = n_read
        for i in tqdm(range(n),desc="Encoding sentences…"):
            self.dataset.append([torch.tensor(tokeniser.encode(dataset[i][0], max_length=1024)), int(dataset[i][1])])
    
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
        IN: pickle file containing full documents, camemBERT tokeniser object,
        and camemBERT model.
        """
        self.dataset = []
        with open(pickle_file, "rb") as file:
            dataset = pickle.load(file)

        if n_read == 'all':
            n = len(dataset)
        else:
            n = n_read
        print("Creating the dataset…")
        for i in tqdm(range(n), desc="Progress"):
            facts, non_facts = dataset[i]['facts'], dataset[i]['non_facts']
            facts = [tokeniser.encode(sentence, max_length=512, truncation_strategy='longest_first') for sentence in facts]
            facts = [torch.tensor(sentence) for sentence in facts]
            non_facts = [tokeniser.encode(sentence, max_length=512, truncation_strategy='longest_first') for sentence in non_facts]
            non_facts = [torch.tensor(sentence) for sentence in non_facts]
            ones, zeros = len(dataset[i]['facts']), len((dataset[i]['non_facts']))
            ones, zeros = torch.ones(ones), torch.zeros(zeros)
            doc = torch.nn.utils.rnn.pad_sequence(facts + non_facts, batch_first=True)
            self.dataset.append([doc, torch.cat([ones, zeros])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class DocumentDataset(Dataset):
    """
    Inherits the PyTorch Dataset class. 
    Reads disc and returns camemBERT document representation
    """
    def __init__(self, ds_loc, ds_len=None, max_length=None):
        """ds_loc : folder containing the raw data"""
        self.loc = ds_loc
        self.max_length = max_length
        self.ds_len = ds_len
    
    def __len__(self):
        if self.ds_len is not None:
            return self.ds_len
        return len(os.listdir(self.loc))
    
    def __getitem__(self, index):
        if self.ds_len is not None and index >= self.ds_len:
            raise IndexError("Index out of range")
        filename = "{}{}.pickle".format(self.loc, str(index))
        doc = []
        with open(filename, "rb") as file:
            while True:
                try:
                    doc.append(pickle.load(file))
                except EOFError:
                    break
        sentences = doc[:-1]
        sentences = [tensor.squeeze(0) for tensor in sentences]
        sentences = torch.stack(sentences)
        annotations = doc[-1]
        if self.max_length is not None:
        	return sentences[:self.max_length], annotations[:self.max_length]
        return sentences, annotations
