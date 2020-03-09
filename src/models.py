"""
Module containing all experimental mod
"""
import torch
from torch import nn


class FactsOrAnalysis(nn.Module):
    def __init__(self,embeddings_tensor,hidden_size=512,dropout=.5,gru_dropout=.3,embedding_size=200,out_channels=100,kernel_size=3,max_sen_len=50):
        super(FactsOrAnalysis, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.max_sen_len = max_sen_len
        self.embedding = nn.Embedding.from_pretrained(embeddings_tensor)
        self.rnn = nn.GRU(embedding_size,
                          hidden_size,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True,
                          dropout=gru_dropout)

        # Taken from the TextCNN implementation by Anubhav Gupta
        # https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_size, out_channels=self.out_channels, kernel_size=self.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size+1)
        )
        
        self.linear = nn.Linear(self.hidden_size+self.out_channels,2)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(dropout)
    
    def initHidden(self):
        n=1 if self.rnn.bidirectional==False else 2
        l=self.rnn.num_layers
        return torch.zeros(n*l, 1, self.hidden_size, device=device)
    
    def forward(self,input_tensor):
        hidden=self.initHidden()
#         print("input:",input_tensor.shape)
        output = self.embedding(input_tensor)
#         print("output:",output.shape)
        conv_output = self.conv1(output.permute(0,2,1))
        conv_output = conv_output.squeeze(2)
        output = self.drop(output)
        _, hidden = self.rnn(output)
        hidden = hidden[0]
        cat_tensors = torch.cat((conv_output,hidden),1)
        cat_tensors = self.drop(cat_tensors)
        output = self.linear(cat_tensors)
        return self.softmax(output)



class FoaCamemBERTrnn(nn.Module):
    """FIRST EXPERIMENT: camemBERT as embeddings model that feeds into a GRU network
Originally a combination of the output of a GRU network and a CNN, I am now replacing
the architecture to include a BERT model whose output is fed into a recurrent layer"""
    def __init__(self, camembert, hidden_size=128):
        super(FoaCamemBERTrnn, self).__init__()
        self.camembert = camembert
        self.gru = torch.nn.GRU(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigma = torch.nn.Sigmoid()

    def forward(self, input_tensor, lengths=None):
        output = self.camembert(input_tensor)[0]
        if lengths is not None:
            output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=True)
        _, hidden = self.gru(output)
        output = hidden[0]
        output = self.linear(output)
        return self.sigma(output)


class FoaCamemBERTlinear(nn.Module):
    """SECOND EXPERIMENT: camemBERT only
    I compare the stand-alone camemBERT model with a linear layer at the top and the
    original architecture"""
    def __init__(self, camembert):
        super(FoaCamemBERTlinear, self).__init__()
        self.camembert = camembert
        self.sigma = nn.Sigmoid()

    def forward(self, input_tensor):
        return self.sigma(self.camembert(input_tensor)[0])

class FoaFlauBERTlinear(nn.Module):
    """THIRD EXPERIMENT: I compare the stand-alone FlauBERT model with a linear layer at the top and
    the original architecture"""
    def __init__(self, flaubert):
        super(FoaFlauBERTlinear, self).__init__()
        self.flaubert = flaubert
        self.sigma = nn.Sigmoid()

    def forward(self, input_tensor):
        return self.sigma(self.flaubert(input_tensor)[0])
