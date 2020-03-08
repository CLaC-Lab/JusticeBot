import torch,IPython,numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from transformers import AdamW

class PadSequence:
    def __call__(self,batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.Tensor([len(x) for x in sequences])
        labels = torch.Tensor([x[1] for x in sorted_batch])
        return sequences_padded, labels, lengths

def train(input_tensor,target,model,optimiser,criterion,clip,lengths):
    model.train()
    optimiser.zero_grad()
    prediction = model(input_tensor,lengths=lengths)
    loss = criterion(prediction.squeeze(1),target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
    optimiser.step()
    return loss.item(), prediction

def valid(input_tensor,target,model,criterion,lengths):
    model.eval()
    prediction = model(input_tensor,lengths=lengths)
    loss = criterion(prediction.squeeze(1),target)
    return loss.item(), prediction

def test(target_tensor,prediction_tensor):
    t = target_tensor.cpu().detach().numpy()
    t = np.array([target for target in t])
    
    p = prediction_tensor.cpu().detach().numpy()
    p = np.array([predic for predic in p])
    p = p.round()
    p = p.squeeze(1)
#     print(t,p)
            
    return accuracy_score(t,p), f1_score(t,p), precision_score(t,p), recall_score(t,p)

def trainIters(model,
               train_dset,
               valid_dset,
               batch_size,
               n_epochs,
               learning_rate,
               weight_decay,
               clip,
               device,
               collate_fn=PadSequence()):
    
    print("CUDA is available!" if torch.cuda.is_available() else "NO CUDA 4 U")
    
    optimiser = AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    
    train_dl=torch.utils.data.DataLoader(train_dset, batch_size=batch_size,collate_fn=collate_fn)
    valid_dl=torch.utils.data.DataLoader(valid_dset, batch_size=batch_size,collate_fn=collate_fn)
    
    train_losses=[np.inf]
    valid_losses=[np.inf]
    train_acc=train_prec=train_rec=0
    valid_acc=valid_prec=valid_rec=0
    tqdm_range=tqdm(range(1,n_epochs+1),desc='Epoch',leave=False)
    for epoch in tqdm_range:
        
        #################################
        ### STUFF RELATED TO PLOTTING ###
        #################################
        plt.gca().cla()
        plt.xlim(0,n_epochs)
        plt.ylim(0,2)
        plt.title("Learning curve")
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.text(n_epochs/2,1.9,"Train loss: {:.2f}".format(train_losses[-1]))
        plt.text(n_epochs/2,1.8,"Validation loss: {:.2f}".format(valid_losses[-1]))
        plt.text(n_epochs/2,1.7,"Tr acc: {:.2f}".format(train_acc))
        plt.text(n_epochs/2,1.6,"Tr pre: {:.2f}".format(train_prec))
        plt.text(n_epochs/2,1.5,"Tr rec: {:.2f}".format(train_rec))
        plt.text(n_epochs/2,1.4,"Va acc: {:.2f}".format(valid_acc))
        plt.text(n_epochs/2,1.3,"Va pre: {:.2f}".format(valid_prec))
        plt.text(n_epochs/2,1.2,"Va rec: {:.2f}".format(valid_rec))
        plt.plot(train_losses, "-b", label="Training loss")
        plt.plot(valid_losses, "-r", label="Validation loss")
        plt.legend(loc="upper left")
        IPython.display.display(plt.gcf())
        ########################################
        ### END OF STUFF RELATED TO PLOTTING ###
        ########################################
        avg_train=[]
        train_acc=[]
        train_prec=[]
        train_rec=[]
        train_dl=tqdm(train_dl,desc='Training',leave=False)
        
#         for i in train_dl:
#             print(i[0],i[1],i[2])
    
        for x,y,lengths in train_dl:
            input_tensor = x.to(device)
            target = y.to(device)
            train_loss, prediction = train(input_tensor,
                                           target,
                                           model,
                                           optimiser,
                                           criterion,
                                           clip,
                                           lengths)
            accuracy,f1,precision,recall=test(target,prediction)
            avg_train.append(train_loss)
            train_acc.append(accuracy)
            train_prec.append(precision)
            train_rec.append(recall)
            train_dl.set_description('Training accuracy: {:.4f}'.format(accuracy))
        avg_train=sum(avg_train)/len(avg_train)
        
        train_acc=sum(train_acc)/len(train_acc)
        train_prec=sum(train_prec)/len(train_prec)
        train_rec=sum(train_rec)/len(train_rec)
        train_losses.append(avg_train)
        
        with torch.no_grad():
            avg_valid=[]
            valid_acc=[]
            valid_prec=[]
            valid_rec=[]
            valid_dl=tqdm(valid_dl,desc='Validating',leave=False)
            for x, y,lengths in valid_dl:
                input_tensor = x.to(device)
                target = y.to(device)
                v_loss, valid_pred = valid(input_tensor,
                                           target,
                                           model,
                                           criterion,
                                           lengths)
                accuracy,f1,precision,recall=test(target,valid_pred)
                avg_valid.append(v_loss)
                valid_acc.append(accuracy)
                valid_prec.append(precision)
                valid_rec.append(recall)
                valid_dl.set_description('Validation accuracy: {:.4f}'.format(accuracy))
            avg_valid=sum(avg_valid)/len(avg_valid)
            valid_acc=sum(valid_acc)/len(valid_acc)
            valid_prec=sum(valid_prec)/len(valid_prec)
            valid_rec=sum(valid_rec)/len(valid_rec)
            valid_losses.append(v_loss)
            
        IPython.display.clear_output(wait=True)
        tqdm_range.refresh()
    
    return train_losses,valid_losses