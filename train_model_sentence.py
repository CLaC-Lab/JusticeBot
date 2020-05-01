"""
Training and evaluation script for the binary sentence classifier
"""
import torch
import gensim
from src.models import SentenceClassifier
from src.dataset import SentenceDataset
from src.train import trainIters, evaluateModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings_file = "../frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True, unicode_errors='ignore')
embeddings_tensor = torch.FloatTensor(embeddings.vectors)
pickle_file = "data/dataset_sentences_facts_non_facts20200427.pickle"
dataset = SentenceDataset(pickle_file=pickle_file, embeddings_file=embeddings_file, n_sentences=0)
tr = int(len(dataset)*.70)
vd = int(len(dataset)*.10)
ts = len(dataset) - tr - vd
train_ds, valid_ds, test_ds = torch.utils.data.random_split(dataset, [tr, vd, ts])
model = SentenceClassifier(embeddings_tensor=embeddings_tensor).to(device)

## CONFIG
batch_size = 512
n_epochs = 30
learning_rate = 1e-4
weight_decay = 0
clip = .2
## /CONFIG

t, v = trainIters(model, 
    train_ds, 
    valid_ds, 
    batch_size, 
    n_epochs, 
    learning_rate, 
    weight_decay, 
    clip, 
    device=device, 
    collate_fn="sentence",
    lr_step=[20])

torch.save(model.state_dict(), "sentence_clf.pt")
model.eval()
evaluateModel(model=model, 
    filename="sentence_clf", 
    test_set=test_ds, 
    batch_size=batch_size,
    device=device,
    collate_fn="sentence")
# avg_acc = []
# avg_prec = []
# avg_rec = []
# avg_f1 = []
# data_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, collate_fn=PadSequence())
# data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
# file = open("sentence_clf.eval", "w")
# file.write("accuracy\tprecision\trecall\t\tf1\n")
# for sentence, annotation, lengths in data_loader:
#     prediction = model(sentence.to(device), lengths)
#     acc, prec, rec, f1 = test(annotation, prediction)
#     avg_acc.append(acc)
#     avg_prec.append(prec)
#     avg_rec.append(rec)
#     avg_f1.append(f1)
# acc = sum(avg_rec)/len(avg_rec)
# prec = sum(avg_prec)/len(avg_prec)
# rec = sum(avg_rec)/len(avg_rec)
# f1 = sum(avg_f1)/len(avg_f1)
# file.write("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(acc, prec, rec, f1))
# file.close()
# print("Done! See the output file for results.\n")
