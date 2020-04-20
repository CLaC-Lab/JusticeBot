"""
Module containing all experimental mod
"""
import torch
from torch import nn

class SentenceClassifier(torch.nn.Module):
    """
    Performs binary sentence classification
    """
    def __init__(self, embeddings_tensor,
                 hidden_size=512, 
                 dropout=.5,
                 embedding_size=200, 
                 out_channels=100, 
                 kernel_size=3, 
                 max_sen_len=50):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings_tensor)
        self.gru = torch.nn.GRU(embedding_size,
                          hidden_size,
                          batch_first=True,
                          bidirectional=True)
        # Taken from the TextCNN implementation by Anubhav Gupta
        # https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=embedding_size, out_channels=out_channels, kernel_size=kernel_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(max_sen_len - kernel_size+1)
        )
        
        self.linear = torch.nn.Linear(in_features=hidden_size+out_channels, out_features=1)
        self.drop = torch.nn.Dropout(dropout)
    
    def forward(self, input, lengths):
        output = self.embedding(input)
        conv_output = self.conv1(output.permute(0, 2, 1))
        conv_output = conv_output.squeeze(2)
        output = self.drop(output)
        output = torch.nn.utils.rnn.pack_padded_sequence(output, lengths, batch_first=True)
        _, hidden = self.gru(output)
        hidden = hidden[0]
        cat_tensors = torch.cat((conv_output, hidden), dim=1)
        cat_tensors = self.drop(cat_tensors)
        output = self.linear(cat_tensors)
        output = torch.sigmoid(output).permute(1,0)
        return output

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
    def __init__(self, embedding_size, hidden_size=128):
        super(GRULinear, self).__init__()
        self.gru = torch.nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(in_features=2*hidden_size, out_features=1)
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
        backward = reversed(backward)
        output = torch.cat([forward, backward], dim=1)
        output = self.output(output)
        return output

class AttentionDecoder(torch.nn.Module):
    """
    Attention decoder that calculates the annotation weights and
    applies the attention mechanism to the decoding step
    """
    def __init__(self, hidden_size, max_len=128):
        """
        Create Linear layer for the alignment model and
        GRU network that decodes the input
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.attn1 = torch.nn.Linear(in_features=hidden_size, out_features=max_len, bias=False)
        self.attn2 = torch.nn.Linear(in_features=max_len, out_features=1, bias=False)
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output = torch.nn.Linear(in_features=hidden_size, out_features=1)
  
    def forward(self, annotations, hidden):
        attn = torch.tanh(self.attn1(annotations))
        attn = torch.softmax(self.attn2(attn), dim=0)
        attn_applied = attn*annotations
        attn_applied = torch.cat([attn_applied, hidden])
        output, hidden = self.gru(attn_applied.unsqueeze(0))
        hidden = hidden.squeeze(0)
        output = output.squeeze(0)
        output = self.output(output)
        output = torch.sigmoid(torch.sum(output, dim=0))
        return output, hidden

class AttEncoderDecoder(torch.nn.Module):
    """
    Encoder-Decoder architecture with an original implementation of the
    attention mechanism (Bahdanau 2015)
    """
    def __init__(self, hidden_size, max_len, device):
        super().__init__()
        self.max_len = max_len
        self.encoder = AttentionEncoder(hidden_size=hidden_size, input_size=64)
        self.decoder = AttentionDecoder(hidden_size=hidden_size, max_len=self.max_len)
        self.hidden_size = hidden_size
        self.device = device
    def forward(self, input):
        annotations = self.encoder(input)
        hidden = torch.zeros([1, self.hidden_size]).to(self.device)
        output = []
        for i in range(len(annotations)):
            sent, hidden = self.decoder(annotations, hidden)
            output.append(sent)
        output = torch.stack(output).permute(1,0)
        return output
