import torch,IPython,numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(input_tensor,target,model,optimiser,criterion,clip):
    model.train()
    optimiser.zero_grad()
    prediction = model(input_tensor)
    loss = criterion(prediction,target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
    optimiser.step()
    return loss.item(), prediction

def valid(input_tensor,target,model,criterion):
    model.eval()
    prediction = model(input_tensor)
    loss = criterion(prediction,target)
    return loss.item(), prediction

def test(target_tensor,prediction_tensor):
    t = target_tensor.cpu().detach().numpy()
    t = np.array([target[0] for target in t])
    
    p = prediction_tensor.cpu().detach().numpy()
    p = np.array([predic[0] for predic in p])
    p = p.round()
            
    return accuracy_score(t,p), f1_score(t,p), precision_score(t,p), recall_score(t,p)

def trainIters(model,
               train_dset,
               valid_dset,
               batch_size,
               n_epochs,
               learning_rate,
               weight_decay,
               clip,
               collate_fn=None):
    
    print("CUDA is available!" if torch.cuda.is_available() else "NO CUDA 4 U")
    
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    criterion = torch.nn.BCELoss()
    
    train_dl=torch.utils.data.DataLoader(train_dset, batch_size=batch_size,collate_fn=collate_fn)
    valid_dl=torch.utils.data.DataLoader(valid_dset, batch_size=batch_size,collate_fn=collate_fn)
    
    train_losses=[np.inf]
    valid_losses=[np.inf]
    train_f1=0
    valid_f1=0
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
        plt.text(n_epochs/2,1.7,"Training F1: {:.2f}".format(train_f1))
        plt.text(n_epochs/2,1.6,"Validation F1: {:.2f}".format(valid_f1))
        plt.plot(train_losses, "-b", label="Training loss")
        plt.plot(valid_losses, "-r", label="Validation loss")
        plt.legend(loc="upper left")
        IPython.display.display(plt.gcf())
        ########################################
        ### END OF STUFF RELATED TO PLOTTING ###
        ########################################
        avg_train=[]
        train_f1=[]
        train_dl=tqdm(train_dl,desc='Training',leave=False)
        for x,y in train_dl:
            input_tensor = x.to(device)
            target = y.to(device)
            train_loss, prediction = train(input_tensor,target,model,optimiser,criterion,clip)
            accuracy,f1,precision,recall=test(target,prediction)
            avg_train.append(train_loss)
            train_f1.append(f1)
            train_dl.set_description('Training accuracy: {:.4f}'.format(accuracy))
        avg_train=sum(avg_train)/len(avg_train)
        train_f1=sum(train_f1)/len(train_f1)
        
        train_losses.append(avg_train)
        
        with torch.no_grad():
            avg_valid=[]
            valid_f1=[]
            valid_dl=tqdm(valid_dl,desc='Validating',leave=False)
            for x, y in valid_dl:
                input_tensor = x.to(device)
                target = y.to(device)
                v_loss, valid_pred = valid(input_tensor,target,model,criterion)
                accuracy,f1,precision,recall=test(target,valid_pred)
                avg_valid.append(v_loss)
                valid_f1.append(f1)
                valid_dl.set_description('Validation accuracy: {:.4f}'.format(accuracy))
            avg_valid=sum(avg_valid)/len(avg_valid)
            valid_f1=sum(valid_f1)/len(valid_f1)
            valid_losses.append(v_loss)
            
        IPython.display.clear_output(wait=True)
        tqdm_range.refresh()
    
    return train_losses,valid_losses