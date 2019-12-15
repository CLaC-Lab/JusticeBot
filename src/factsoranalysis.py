import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FactsOrAnalysis(nn.Module):
    def __init__(self,embeddings_tensor,hidden_size=512,dropout=.5,gru_dropout=.3,embedding_size=200,
                out_channels=100,
                kernel_size=3,
                max_sen_len=50):
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
#         print("HOAL")
        output = self.embedding(input_tensor)
        conv_output = self.conv1(output.permute(0,2,1))
        conv_output = conv_output.squeeze(2)
        output = self.drop(output)
        _, hidden = self.rnn(output)
        hidden = hidden[0]
        cat_tensors = torch.cat((conv_output,hidden),1)
        cat_tensors = self.drop(cat_tensors)
        output = self.linear(cat_tensors)
        return self.softmax(output)