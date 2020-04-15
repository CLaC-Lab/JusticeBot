import torch, IPython, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from transformers import AdamW

class PadSequence:
    """
    Collate function that returns padded sequence to pass to DataLoader
    """
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        lengths = torch.LongTensor([len(x) for x in sequences])
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        labels = torch.Tensor([x[1] for x in sorted_batch])
        return sequences_padded, labels.unsqueeze(0), lengths

def train(input_tensor, target, model, optimiser, criterion, clip, lengths):
    model.train()
    optimiser.zero_grad()
    if lengths is None:
        prediction = model(input_tensor)
    else:
        prediction = model(input_tensor, lengths)
    loss = criterion(prediction, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimiser.step()
    return loss.item(), prediction

def valid(input_tensor, target, model, criterion, lengths):
    model.eval()
    if lengths is None:
        prediction = model(input_tensor)
    else:
        prediction = model(input_tensor, lengths)
    loss = criterion(prediction, target)
    return loss.item(), prediction

def test(target_tensor, prediction_tensor):
    t = target_tensor.cpu().detach().numpy()
    t = np.array([target for target in t])

    p = prediction_tensor.cpu().detach().numpy()
    p = np.array([predic for predic in p])
    p = p.round()
    # print("\nt, p shapes: {}, {}".format(t.shape, p.shape))
    t, p = t.squeeze(0), p.squeeze(0)
    try:
        acc = accuracy_score(t, p)
        pre = precision_score(t, p)
        rec = recall_score(t, p)
    except ValueError:
        print("ValueError!")
        print("t, p: {}, {}".format(t, p))
        print("Skipping this training example...")
        acc = pre = rec = 0
    return acc, pre, rec

def trainIters(model,
               train_dset,
               valid_dset,
               batch_size,
               n_epochs,
               learning_rate,
               weight_decay,
               clip,
               device,
               collate_fn=None):

    with open("output", "w") as op:
        op.write("tr loss\ttr acc\ttr prec\ttr rec\tv loss\tv acc\tv prec\tv rec\n")
        
    print("CUDA is available!" if torch.cuda.is_available() else "NO CUDA 4 U")
    
    optimiser = AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    if collate_fn == "sentence":
        col_func = PadSequence()
    else: col_func = None
    train_dl = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, collate_fn=col_func)
    valid_dl = torch.utils.data.DataLoader(valid_dset, batch_size=batch_size, collate_fn=col_func)
    
    train_losses = [np.inf]
    valid_losses = [np.inf]
    train_acc = train_prec = train_rec = 0
    valid_acc = valid_prec = valid_rec = 0
    tqdm_range = tqdm(range(1, n_epochs+1), desc='Epochs', leave=False)
    first_epoch = True
    for epoch in tqdm_range:
        
        #################################
        ### STUFF RELATED TO PLOTTING ###
        #################################
        # plt.gca().cla()
        # plt.xlim(0,n_epochs)
        # plt.ylim(0,2)
        # plt.title("Learning curve")
        # plt.xlabel("Number of epochs")
        # plt.ylabel("Loss")
        # plt.text(n_epochs/2,1.9,"Train loss: {:.2f}".format(train_losses[-1]))
        # plt.text(n_epochs/2,1.8,"Validation loss: {:.2f}".format(valid_losses[-1]))
        # plt.text(n_epochs/2,1.7,"Tr acc: {:.2f}".format(train_acc))
        # plt.text(n_epochs/2,1.6,"Tr pre: {:.2f}".format(train_prec))
        # plt.text(n_epochs/2,1.5,"Tr rec: {:.2f}".format(train_rec))
        # plt.text(n_epochs/2,1.4,"Va acc: {:.2f}".format(valid_acc))
        # plt.text(n_epochs/2,1.3,"Va pre: {:.2f}".format(valid_prec))
        # plt.text(n_epochs/2,1.2,"Va rec: {:.2f}".format(valid_rec))
        # plt.plot(train_losses, "-b", label="Training loss")
        # plt.plot(valid_losses, "-r", label="Validation loss")
        # plt.legend(loc="upper left")
        # IPython.display.display(plt.gcf())
        ########################################
        ### END OF STUFF RELATED TO PLOTTING ###
        ########################################
        avg_train = []
        train_acc = []
        train_prec = []
        train_rec = []
        train_dl = tqdm(train_dl, desc='Training examples', leave=False)
        
#         for i in train_dl:
#             print(i[0],i[1],i[2])
    
        for datum in train_dl:
            input_tensor = datum[0].to(device)
            target = datum[1].to(device)
            if collate_fn is None:
                lengths = None
            else:
                lengths = datum[2]
            train_loss, prediction = train(input_tensor,
                                       target,
                                       model,
                                       optimiser,
                                       criterion,
                                       clip,
                                       lengths)
            accuracy, precision, recall = test(target, prediction)
            avg_train.append(train_loss)
            train_acc.append(accuracy)
            train_prec.append(precision)
            train_rec.append(recall)
            train_dl.set_description("Training loss: {:.4f}".format(train_loss))
        avg_train = sum(avg_train)/len(avg_train)
        
        train_acc = sum(train_acc)/len(train_acc)
        train_prec = sum(train_prec)/len(train_prec)
        train_rec = sum(train_rec)/len(train_rec)
        train_losses.append(avg_train)
        with torch.no_grad():
            avg_valid = []
            valid_acc = []
            valid_prec = []
            valid_rec = []
            valid_dl = tqdm(valid_dl, desc='Validation examples', leave=False)
            for datum in valid_dl:
                input_tensor = datum[0].to(device)
                target = datum[1].to(device)
                if collate_fn is None:
                    lengths = None
                else:
                    lengths = datum[2]
                v_loss, valid_pred = valid(input_tensor,
                                           target,
                                           model,
                                           criterion,
                                           lengths)
                accuracy, precision, recall = test(target, valid_pred)
                avg_valid.append(v_loss)
                valid_acc.append(accuracy)
                valid_prec.append(precision)
                valid_rec.append(recall)
                valid_dl.set_description('Validation loss: {:.4f}'.format(v_loss))
            avg_valid = sum(avg_valid)/len(avg_valid)
            valid_acc = sum(valid_acc)/len(valid_acc)
            valid_prec = sum(valid_prec)/len(valid_prec)
            valid_rec = sum(valid_rec)/len(valid_rec)
            valid_losses.append(v_loss)
            
        # IPython.display.clear_output(wait=True)
        # tqdm_range.refresh()
        with open("output", "a") as op:
            op.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t\n".format(avg_train,train_acc,train_prec,train_rec,avg_valid,valid_acc,valid_prec,valid_rec))
    return train_losses, valid_losses
