# First run of experiments: camemBERT as embeddings model

My first approach to camemBERT was to use it as an embeddings model and attach it to an GRU network. I removed the CNN part of of the encoder because of ~~lazyness~~ technical difficulties of implementing the new architecture and the constrain of having a fixed-length sequence model.