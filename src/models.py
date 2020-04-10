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

class FactOrAnalysisRNN(nn.Module):
    """FOURTH EXPERIMENT: seq2seq from text to binary string
    I take a document, encode it as a sequence of byte-pair sequences, pass them through
    camemBERT, and then through a GRU network
    """
    def __init__(self, camembert, hidden_size=512):
        super(FactOrAnalysisRNN, self).__init__()
        self.camembert = camembert
        self.gru = nn.GRU(768, hidden_size, bidirectional=True)
        self.sigma = nn.Sigmoid()

    def forward(self, tensor):
        print(tensor.shape)
        print(tensor[0].shape)
        tensor = self.camembert(tensor[0])[1]
        tensor = tensor.unsqueeze(0)
        print(tensor.shape)
        tensor = self.gru(tensor)[0][:,:,-1]
        return self.sigma(tensor).squeeze(0)

class LSTMLinear(torch.nn.Module):
    """
    LSTM network with a linear head that takes in a camemBERT
    document representation and returns a binary sequence
    """
    def __init__(self, embedding_size):
        super(LSTMLinear, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_size=64, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(in_features=2*embedding_size, out_features=1)
        self.sigma = torch.nn.Sigmoid()
    def forward(self, x):
        output = self.lstm(x)[0]
        output = self.linear(output)
        output = self.sigma(output)
        return output.squeeze(2)

class GRULinear(torch.nn.Module):
    """
    GRU network with a linear head that takes in a camemBERT
    document representation and returns a binary sequence
    """
    def __init__(self, embedding_size):
        super(GRULinear, self).__init__()
        self.gru = torch.nn.GRU(input_size=embedding_size, hidden_size=64, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(in_features=2*embedding_size, out_features=1)
        self.sigma = torch.nn.Sigmoid()
    def forward(self, x):
        output = self.gru(x)[0]
        output = self.linear(output)
        output = self.sigma(output)
        return output.squeeze(2)

class AttentionEncoder(torch.nn.Module):
    """
    Attention encoder that returns the annotation representation
    of the input sequence
    """
    def __init__(self, hidden_size, input_size):
        """
        Create the rnn cell object
        """
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.output = torch.nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.hidden_size = hidden_size
    def forward(self, input):
        """
        Iterate over the input in both directions gathering
        hidden states and concatenate them to create annotations
        """
        output = self.gru(input)[0]
        output = output.view(len(input[0]), len(input), 2, self.hidden_size).squeeze(1)
        forward = output[:,0,:]
        backward = output[:,1,:]
        output = torch.cat([forward, backward], dim=1)
        output = self.output(output)
        return output

class AttentionDecoder(torch.nn.Module):
    """
    Attention decoder that calculates the annotation weights and
    applies the attention mechanism to the decoding step
    """
    def __init__(self, hidden_size, max_length=256):
        """
        Create Linear layer for the alignment model and
        GRU network that decodes the input
        """
        super().__init__()
        self.attn = torch.nn.Linear(in_features=hidden_size, out_features=max_length)
        self.max_len = max_length
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=1)
        self.output = torch.nn.Linear(in_features=2, out_features=1)
        
    def forward(self, annotations, hidden, prev_sent):
        """
        Combine encoder annotations with hidden state to obtain
        attention weights and use these to create the i-th context
        vector
        """
        attn = torch.cat([annotations, hidden])
        attn = torch.nn.functional.softmax(self.attn(attn), dim=1)
        annotations = torch.nn.functional.pad(annotations, (0,0,0,self.max_len-len(annotations)))
        context = torch.bmm(attn.unsqueeze(0), annotations.unsqueeze(0)).squeeze(0)
        attn_input = torch.cat([hidden, context], dim=0).unsqueeze(0)
        output, hidden = self.gru(attn_input)
        output = self.linear(output).permute(0,2,1)
        output = torch.cat([torch.mean(output).unsqueeze(0), prev_sent.unsqueeze(0)])
        output = torch.sigmoid(self.output(output))
        return output.squeeze(0), hidden.squeeze(0)

class AttEncoderDecoder(torch.nn.Module):
    """
    Encoder-Decoder architecture with an original implementation of the
    attention mechanism (Bahdanau 2015)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = AttentionEncoder(hidden_size=hidden_size, input_size=64)
        self.decoder = AttentionDecoder(hidden_size=hidden_size)
        self.hidden_size = hidden_size
    def forward(self, input):
        annotations = self.encoder(input)
        prev_sent = torch.tensor(1.0)
        hidden = torch.zeros([1, self.hidden_size])
        output = []
        for i in range(len(annotations)):
            prev_sent, hidden = self.decoder(annotations, hidden, prev_sent)
            output.append(prev_sent)
        output = torch.stack(output)
        return output