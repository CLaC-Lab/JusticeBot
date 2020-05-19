"""General utility functions"""
import torch
def getBestSplit(binary_sequence, gamma=1):
    """
    returns best split between facts and analysis sections
    in : list of integers for candidate indices such as [1,1,0,1,0] for instance
    out : integer, position of the last paragraph in be included in the fact section
    """
    purities = {}
    for i in range(1, len(binary_sequence)+1):
        substring = binary_sequence[:i]
        purity = sum(substring)/len(substring)
        purities[len(substring)] = purity
#     print(purities)
    purities = {key:value for key, value in purities.items() if value != 1}
    index = max(purities, key=lambda key: purities[key])
    index = int(index*gamma)
    return binary_sequence[:index]

def padSentence(sentence, max_len=50):
    '''
    Pads tensor representation of sentence with zeros
    '''
    padding = max_len - len(sentence)
    sentence = torch.nn.functional.pad(sentence, pad=(0, padding), mode='constant', value=0)
    return sentence

def getOffsets(offsets):
    of = {
    "<-4" : 0,
    "-4" : 0,
    "-3" : 0,
    "-2" : 0,
    "-1" : 0,
    "0" : 0,
    "1" : 0,
    "2" : 0,
    "3" : 0,
    "4" : 0,
    ">4" : 0}
    for n in offsets:
        if n == 0: of["0"] += 1
        elif n == 1: of["1"] += 1
        elif n == 2: of["2"] += 1
        elif n == 3: of["3"] += 1
        elif n == 4: of["4"] += 1
        elif n > 4: of[">4"] += 1
        elif n == -1: of["-1"] += 1
        elif n == -2: of["-2"] += 1
        elif n == -3: of["-3"] += 1
        elif n ==-4: of["-4"] += 1
        elif n < -4: of["<-4"] += 1
    return of

class Reduced(torch.nn.Module):
    def __init__(self,camembert):
        super(Reduced,self).__init__()
        self.camembert = camembert
        self.reduce = torch.nn.Linear(in_features=768, out_features=64)
        
    def forward(self,x):
        x = self.camembert(x)[1]
        x = self.reduce(x)
        return x
